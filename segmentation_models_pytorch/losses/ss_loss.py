class Deep_Supervised_Loss(nn.Module):
    def __init__(self):

        super().__init__()
        self.loss = smp.losses.TverskyLoss(mode="binary",from_logits=False,)

    def forward(self, input, target):
        loss = 0
        # print(type(input))
        for i, img in enumerate(input):
            w = 1 / (2 ** i)
            
            label = F.interpolate(target,size=img.shape[2:])

            l = self.loss(torch.sigmoid(img), label)
            
            loss += l * w
            
        return loss    
