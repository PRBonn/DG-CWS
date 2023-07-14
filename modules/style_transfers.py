import torch


def transfer_global_statistics(src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
  """ Transfer the statistics from the src tensor to the target tensor.

  Accordingly the output tensor has the statistics of the src tensor.

  Args:
      src (torch.Tensor): Source tensor of shape [B x C x H1 x W1]
      trg (torch.Tensor): Target tensor of shape [B x C x H2 x W2] 

  Returns:
      torch.Tensor: Target tensor with statistics of src tensor.
  """ 
  assert (not src.isnan().any()), "Invalid input the 'src' tensor contains 'nan' values."
  assert (not trg.isnan().any()), "Invalid input the 'trg' tensor contains 'nan' values."

  batch_size, chans, _, _ = src.shape
  trg_height, trg_width = trg.shape[2:]

  src_flattened = src.view(batch_size, chans, -1) # [B x C x H1*W1]
  trg_flattened = trg.view(batch_size, chans, -1) # [B x C x H2*W2]
  
  src_mean = torch.mean(src_flattened, dim=-1, keepdim=True) # [B, C, 1]
  trg_mean = torch.mean(trg_flattened, dim=-1, keepdim=True) # [B, C, 1]

  src_reduced = src_flattened - src_mean # [B x C x H1*W1]
  trg_reduced = trg_flattened - trg_mean # [B x C x H2*W2]

  src_cov_mat = torch.bmm(src_reduced, src_reduced.transpose(1, 2)) / (src_reduced.shape[-1] - 1)# [B x C x C]
  trg_cov_mat = torch.bmm(trg_reduced, trg_reduced.transpose(1, 2)) / (trg_reduced.shape[-1] - 1) # [B x C x C]
  
  src_eigvals, src_eigvecs = torch.linalg.eigh(src_cov_mat) # eigval -> [B, C], eigvecs -> [B, C, C]
  src_eigvals = torch.clamp(src_eigvals, min=1e-8, max=float(torch.max(src_eigvals))) # valid op since covmat is positive (semi-)definit
  src_eigvals_sqrt = torch.sqrt(src_eigvals).unsqueeze(2) # [B, C, 1]

  trg_eigvals, trg_eigvecs = torch.linalg.eigh(trg_cov_mat) # eigval -> [B, C], eigvecs -> [B, C, C]
  trg_eigvals = torch.clamp(trg_eigvals, min=1e-8, max=float(torch.max(trg_eigvals))) # valid op since covmat is positive (semi-)definit
  trg_eigvals_sqrt = torch.sqrt(trg_eigvals).unsqueeze(2) # [B, C, 1]

  # transfer color statistics form source to target
  W_trg = torch.bmm(trg_eigvecs, (1 / trg_eigvals_sqrt) * trg_eigvecs.transpose(1, 2))
  trg_white = torch.bmm(W_trg, trg_reduced)

  W_src_inv = torch.bmm(src_eigvecs, src_eigvals_sqrt * src_eigvecs.transpose(1, 2))
  trg_transformed = torch.bmm(W_src_inv, trg_white) + src_mean

  trg_transformed = trg_transformed.view(batch_size, chans, trg_height, trg_width)

  alpha = torch.rand((batch_size, 1, 1, 1), device=trg_transformed.device)
  alpha = torch.clamp(alpha, min=0.0, max=0.95)

  trg_transformed = (alpha * trg) + ((1 - alpha) * trg_transformed)

  return trg_transformed