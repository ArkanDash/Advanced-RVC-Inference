def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += (rl.float() - gl.float()).abs().mean()

    return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses, g_losses = [], []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = ((1 - dr) ** 2).mean()
        g_loss = (dg**2).mean()
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = ((1 - dg.float()) ** 2).mean()
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

def discriminator_loss_scaled(disc_real, disc_fake, scale=1.0):
    """Scaled discriminator loss (from Applio).

    Downweights losses from sub-discriminators beyond the midpoint by `scale`.
    This prevents multi-resolution sub-discriminator heads (v3) from dominating
    the total loss, which can improve training stability and quality.
    """
    midpoint = len(disc_real) // 2
    r_losses, g_losses = [], []
    loss = 0
    for i, (d_real, d_fake) in enumerate(zip(disc_real, disc_fake)):
        real_loss = (1 - d_real.float()).pow(2).mean()
        fake_loss = d_fake.float().pow(2).mean()
        total_loss = real_loss + fake_loss
        if i >= midpoint:
            total_loss = total_loss * scale
            real_loss = real_loss * scale
            fake_loss = fake_loss * scale
        loss += total_loss
        r_losses.append(real_loss.item())
        g_losses.append(fake_loss.item())
    return loss, r_losses, g_losses

def generator_loss_scaled(disc_outputs, scale=1.0):
    """Scaled generator loss (from Applio).

    Downweights losses from sub-discriminators beyond the midpoint by `scale`.
    This prevents multi-resolution sub-discriminator heads (v3) from dominating
    the total loss, which can improve training stability and quality.
    """
    midpoint = len(disc_outputs) // 2
    gen_losses = []
    loss = 0
    for i, d_fake in enumerate(disc_outputs):
        loss_value = (1 - d_fake.float()).pow(2).mean()
        if i >= midpoint:
            loss_value = loss_value * scale
        gen_losses.append(loss_value)
        loss += loss_value
    return loss, gen_losses

def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * (-2.0 * logs_p).exp()

    return (kl * z_mask).sum() / z_mask.sum()
