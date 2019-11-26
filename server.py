#!/usr/bin/env python

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
import sys

sys.path.append("../")
from pygen.interface import Example
from pygen.interface import ttypes

import torchvision.models as models
import torch
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import time
import io

torch.manual_seed(41)


class ExampleHandler:
    BASE_MODEL = None

    def instantiate_model(self):
        self.BASE_MODEL = models.vgg19(pretrained=True)

    def make_prediction(self, arr_bytes):
        # Rebuild the prediction array from client raw data
        image = Image.open(io.BytesIO(arr_bytes))
        scale_crop = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224)]
        )

        # The normalization that was applied to the data when alexnet was trained
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if len(image.getbands()) != 3:
            image = image.convert("RGB")

        # Put scaling, cropping, normalzing and converting from PIL image to pytorch into one package
        preprocess = transforms.Compose([scale_crop, transforms.ToTensor(), normalize])

        img_t = preprocess(image)
        batch_t = torch.unsqueeze(img_t, 0)

        initial_time = time.time()
        # Realize prediction on client data
        with torch.no_grad():
            probs = F.softmax(self.BASE_MODEL.forward(batch_t))
            top_prob, pred = torch.topk(probs, 5)
        end_time = time.time() - initial_time
        return ttypes.results(top_prob.tolist()[0], pred.tolist()[0], end_time)


handler = ExampleHandler()
processor = Example.Processor(handler)
transport = TSocket.TServerSocket(port=8000)
tfactory = TTransport.TBufferedTransportFactory()
pfactory = TBinaryProtocol.TBinaryProtocolFactory()

server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

print("Starting python server...")
server.serve()
print("done!")
