import math

import torch
import torch.nn as nn
import numpy as np
from utils.diffusion_util import GLD


class DualDiffusion(nn.Module):
    def __init__(self, config, device, target_dim=96):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.finger_feature_embedding = nn.Linear(3, 3) # 3->3

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = GLD(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )
        elif config_diff["schedule"] == "cosine":
            self.beta = self.betas_for_alpha_bar(
                self.num_steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.5):
        # """
        # Create a beta schedule that discretizes the given alpha_t_bar function,
        # which defines the cumulative product of (1-beta) over time from t = [0,1].
        # :param num_diffusion_timesteps: the number of betas to produce.
        # :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
        #                   produces the cumulative product of (1-beta) up to that
        #                   part of the diffusion process.
        # :param max_beta: the maximum beta to use; use values lower than 1 to
        #                  prevent singularities.
        # """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
            self, observed_data, cond_mask, side_info, is_train,observed_data_latent
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, side_info, is_train,observed_data_latent, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
            self, observed_data, cond_mask, side_info, is_train, observed_data_latent, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t].to(self.device)  # (B,1,1)
        noise = torch.randn_like(observed_data).to(self.device)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = 1 - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

        ####################################

        noise_latent = torch.randn_like(observed_data_latent).to(self.device)
        noisy_data_latent = (current_alpha ** 0.5) * observed_data_latent + (1.0 - current_alpha) ** 0.5 * noise_latent

        total_input_latent = self.set_input_to_diffmodel(noisy_data_latent, observed_data_latent, cond_mask)

        predicted_latent = self.diffmodel(total_input_latent, side_info, t)  # (B,K,L)

        target_mask = 1 - cond_mask
        residual_latent = (noise_latent - predicted_latent) * target_mask
        num_eval = target_mask.sum()
        loss_latent = (residual_latent ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss+loss_latent

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = (current_sample * (1 - cond_mask) + observed_data * cond_mask).detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (observed_data,observed_tp,gt_mask,observed_data_latent) = self.process_data(batch)
        cond_mask = gt_mask

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, side_info, is_train, observed_data_latent)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_tp,
            gt_mask,
            observed_data_latent
        ) = self.process_data(batch)
        # print(np.shape(observed_data))
        # print(np.shape(observed_data))
        # print(np.shape(observed_data))
        pose_t = batch["pose"].to(self.device).float()
        batch_size, total_input_n , joints_n = pose_t.size()
        temp_input_n=10
        temp_output_n = total_input_n-temp_input_n


        with torch.no_grad():
            pose_t_output = pose_t[:, temp_input_n:, :]
            pose_t_output = pose_t_output.view(-1, temp_output_n, 21, 3).cpu().numpy()
            cond_mask = gt_mask
            target_mask = 1 - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)
            # samples= samples.permute(0, 2, 1)
            samples_latent = self.impute(observed_data_latent, cond_mask, side_info, n_samples)
            # samples_latent = samples_latent.permute(0, 2, 1)
            samples_mean = np.mean(samples.cpu().numpy(), axis=1)
            samples_latent_mean = np.mean(samples_latent.cpu().numpy(), axis=1)
            samples_mean_output = samples_mean[:,:,temp_input_n:]
            samples_mean_output = samples_mean_output.transpose(0, 2, 1).reshape(-1, temp_output_n, 21, 3)
            # print(np.shape(samples_mean_output))
            samples_latent_mean_output = samples_latent_mean[:, :, temp_input_n:]
            samples_latent_mean_output = samples_latent_mean_output.transpose(0, 2, 1).reshape(-1, temp_output_n, 21, 3)

            best_samples = samples
            # print(np.shape(pose_t_output), np.shape(samples_mean_output))
            if self.mpjpe_error(torch.from_numpy(samples_mean_output),torch.from_numpy(pose_t_output)) > self.mpjpe_error(torch.from_numpy(samples_latent_mean_output),torch.from_numpy(pose_t_output)):
                best_samples = samples_latent
            samples =best_samples
            # trenorm_pose.view(-1, output_n, args.joints, 3)
        # print(224, np.shape(samples),'post truth',np.shape(pose_t),'mean',np.shape(samples_mean)) #torch.Size([20, 5, 63, 15]) post truth torch.Size([20, 15, 63]) mean (20, 63, 15)

        return samples, observed_data, target_mask, observed_tp

    def process_data(self, batch):
        latent_finger_63 = self.finger_features_process_data(batch)
        pose_latent = latent_finger_63 # [32, 15, 66]
        pose = batch["pose"].to(self.device).float() # [32, 15, 66]
        tp = batch["timepoints"].to(self.device).float()
        mask = batch["mask"].to(self.device).float()
        # print(216, np.shape(pose_t),np.shape(pose), np.shape(tp), np.shape(mask))

        pose_latent = pose_latent.permute(0, 2, 1)
        pose = pose.permute(0, 2, 1)
        mask = mask.permute(0, 2, 1)

        return (
            pose,
            tp,
            mask,
            pose_latent
        )
    def finger_features_process_data(self,batch):
        jt_idx = [[0,1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
        pose = batch["pose"].to(self.device).float()
        batch_sise, input_feature_n, dimension_joints = pose.size()
        reshape_pose = pose.reshape(batch_sise, input_feature_n, 21, 3)
        finger = []
        for i in range(0,5):
            temp_reshape_pose = self.finger_feature_embedding(reshape_pose[:, :, jt_idx[i], :])
            finger.append(temp_reshape_pose) #[32, 15, 4, 3]

        latent_finger = torch.cat([finger[0], finger[1],finger[2],finger[3],finger[4]], dim=2)
        reshaped_latent_finger = torch.reshape(latent_finger, (batch_sise, input_feature_n, -1))
        # latent_finger_63 = self.fc1(reshaped_latent_finger)
        latent_finger_63 = reshaped_latent_finger


        # print(243, np.shape(latent_finger_63)) #[32, 15, 63]
        return  latent_finger_63

    def mpjpe_error(self, batch_imp, batch_gt):
        batch_imp = batch_imp.contiguous().view(-1, 3)
        batch_gt = batch_gt.contiguous().view(-1, 3)

        return torch.mean(torch.norm(batch_gt - batch_imp, 2, 1))

# import torch
# from thop import profile
#
# if __name__ == '__main__':
#     config = {
#         'train':
#             {
#                 'epochs': 100,
#                 'batch_size': 32,
#                 'batch_size_test': 32,
#                 'lr': 1.0e-3
#             },
#         'diffusion':
#             {
#                 'layers': 12,
#                 'channels': 64,
#                 'nheads': 8,
#                 'diffusion_embedding_dim': 128,
#                 'beta_start': 0.0001,
#                 'beta_end': 0.5,
#                 'num_steps': 50,
#                 'schedule': "cosine"
#             },
#         'model':
#             {
#                 'is_unconditional': 0,
#                 'timeemb': 128,
#                 'featureemb': 16
#             }
#     }
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print('Using device: %s' % device)
#
#     model = ModelMain(config, device, target_dim=(21 * 3)).to(device)
#     torch.Size([32, 15, 63])
#     torch.Size([32, 15])
#     torch.Size([32, 15, 63])
#
#     pose = torch.zeros(32,15,63).to(device)
#     mask=  torch.zeros(32,15,63).to(device)
#     timepoints = torch.zeros(32,15).to(device)
#     input_n= 75
#     s = {
#         "pose":pose[:, :input_n + 5],
#         "mask": mask[:, :input_n + 5],
#         "timepoints": timepoints[:, :input_n + 5]
#     }
#     print(271, np.shape(pose[:, :input_n + 5]))
#
#     output = model.evaluate(s, 5)
#     samples, _, eval_impute, _ = output
#     print(274, np.shape(samples))
# # #############################################
#     input_tensor = torch.randn(1, 10, 63)  # Example input tensor
#
#     flops, params = profile(model, inputs=(s,))
#     print(f"FLOPs: {flops}, Parameters: {params}")
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total Parameters: {total_params}")
