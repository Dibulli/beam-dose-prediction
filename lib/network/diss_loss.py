def dice_loss_fun(channel, eps=1e-6):
    def sub_dice_loss(input, target):
        iflat = input[:, channel, :, :].reshape(-1, 1)
        tflat = target[:, channel, :, :].reshape(-1, 1)
        intersection = (iflat * tflat).sum()
        return 1 - (2. * intersection + eps) / (iflat.sum() + tflat.sum() + eps)
    return sub_dice_loss
