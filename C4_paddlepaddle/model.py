import paddle

class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()

        self.resnet = paddle.vision.models.resnet50(pretrained=True)
        self.fc_retrieval = paddle.nn.Linear(1000, 20)

        self.att = paddle.nn.Linear(1000, 1)
        self.fc_identify = paddle.nn.Linear(1000, 2)

    def forward(self, x):
        x = self.resnet(x)
        retrieval = self.fc_retrieval(x)

        att = paddle.transpose(self.att(x), perm=(1, 0))
        x = paddle.matmul(att, x)
        identify = self.fc_identify(x)

        return retrieval, identify