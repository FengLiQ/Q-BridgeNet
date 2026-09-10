import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):  # x: (B, T, D) -> B, T/4, D
        return self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)


class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):  # x: (B, T/4, D) -> B, T, D
        return self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)


class RVQ(nn.Module):
    def __init__(self, encoder, vq_embedding0, vq_embedding1, decoder):
        super().__init__()
        self.encoder = encoder
        self.vq_embedding0 = vq_embedding0
        self.vq_embedding1 = vq_embedding1
        self.decoder = decoder

    def encode(self, x):  # x->z_t,index
        z_t = self.encoder(x)
        q0, idx0 = self.quantize(z_t, self.vq_embedding0)  # q1 (B,D)

        r = z_t - q0

        q1, idx1 = self.quantize(r, self.vq_embedding1)

        z_q = q0 + q1
        return z_t, z_q, idx0, idx1

    def decode(self, token0, token1):
        emb0 = self.vq_embedding0(token0)
        emb1 = self.vq_embedding1(token1)
        z = emb0 + emb1
        x_recon = self.decoder(z)
        return x_recon

    def quantize(self, z, codebook):
        B, L, D = z.shape
        z_ = z.contiguous().view(B * L, D)
        w = codebook.weight
        z_sq = torch.sum(z_ ** 2, dim=1, keepdim=True)  # (B,1)
        w_sq = torch.sum(w ** 2, dim=1)  # (K,)
        dist = z_sq - 2 * (z_ @ w.t()) + w_sq.unsqueeze(0)  # (B, K)
        indices = torch.argmin(dist, dim=1)  # (B,)
        quantized = codebook(indices)  # (B, D)
        return quantized.view(B, L, D), indices.view(B, L)

    def forward(self, x):
        z_t, z_q, idx0, idx1 = self.encode(x)
        e_latent_loss = torch.nn.functional.mse_loss(z_t, z_q.detach())
        q_latent_loss = torch.nn.functional.mse_loss(z_q, z_t.detach())
        vq_loss = q_latent_loss + 0.25 * e_latent_loss
        quantized = z_t + (z_q - z_t).detach()
        x_recon = self.decoder(quantized)
        recon_loss = torch.nn.functional.mse_loss(x_recon, x)
        return x_recon, recon_loss, vq_loss
