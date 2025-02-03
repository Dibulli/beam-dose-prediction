import torch

def focal_loss_factory(channel, gamma=2):
    alpha = [0.25, 0.75]
    smooth = 1e-6

    def sub_focal_loss(pred, target):
        prob = pred[:, channel, :, :]
        prob = torch.clamp(prob, smooth, 1.0 - smooth)
        # print(prob)
        # print(prob)
        pos_mask = (target[:, channel, :, :] == 1).float()
        neg_mask = (target[:, channel, :, :] == 0).float()

        pos_loss = -alpha[0] * torch.pow(torch.sub(1.0, prob), gamma) * torch.log(prob) * pos_mask
        neg_loss = -alpha[1] * torch.pow(prob, gamma) * torch.log(torch.sub(1.0, prob)) * neg_mask
        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
        # print('number of positive', num_pos, 'sum of positive loss', pos_loss)
        num_neg = neg_mask.view(neg_mask.size(0), -1).sum()
        # print('number of negative', num_neg, 'sum of negative loss', neg_loss)

        if num_pos == 0:
            loss = neg_loss/num_neg
            # print('---------only negative----------')
        elif num_neg == 0:
            loss = pos_loss/num_pos
            # print('---------only positive----------')
        else:
            loss = pos_loss / num_pos + neg_loss / num_neg
            # print('---------positive and negative----------')
        return loss

    return sub_focal_loss
