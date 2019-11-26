#!/usr/bin/env python

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
import sys

sys.path.append("../")
from pygen.interface import Example
from pygen.interface import ttypes

import argparse
from PIL import Image, ImageFilter
from torchvision import transforms
import torch
import torchvision.models as models
import requests
import torch.nn.functional as F
import prettytable
import io, os, time
import csv
import random
import matplotlib.pyplot as plt
import numpy as np

# using docker
# SERVER_IP = os.environ['CLIENT_SERVER']

# runing locally
# SERVER_IP = 'localhost'
# SERVER_PORT = 8000

SERVER_IP = "34.95.219.7"
SERVER_PORT = 5000

LOWER_THRESHOLD = 0.5
UPPER_THRESHOLD = 0.7

torch.manual_seed(41)
random.seed(100)

def load_image(path_image, noisy):
    img = Image.open(path_image)
    img = img.filter(ImageFilter.BLUR) if noisy else img

    # Set up a transform that scales and crops an image so it has the dimensions
    # of the input layer of alexnet
    scale_crop = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224)]
    )

    # The normalization that was applied to the data when alexnet was trained
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if len(img.getbands()) != 3:
        img = img.convert("RGB")
    # Put scaling, cropping, normalzing and converting from PIL image to pytorch into one package
    preprocess = transforms.Compose([scale_crop, transforms.ToTensor(), normalize])

    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t


def beutify_table(labels, preds, top_probs):
    table = prettytable.PrettyTable(["Label", "Prob"])

    for label, prob in zip(
        map(lambda idx: labels[idx], preds.view(-1).cpu().numpy()),
        top_probs.view(-1).data.cpu().numpy(),
    ):
        table.add_row([label, prob])
    print(table)


def instantiate_model():
    client.instantiate_model()
    model = models.alexnet(pretrained=True)
    return model


def server_request(path_image, labels):
    img = Image.open(path_image)
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format="PNG")
    byte_im = imgByteArr.getvalue()
    size_bytes = sys.getsizeof(byte_im)
    server_response = client.make_prediction(byte_im)
    return server_response, size_bytes


def plot_diff_probs(client_probs, server_probs):
    client_probs = np.array(client_probs)
    server_probs = np.array(server_probs)
    plt.plot(range(1, len(client_probs) + 1), client_probs, label="Cliente")
    plt.plot(range(1, len(client_probs) + 1), server_probs, label="Servidor")
    plt.xlabel("Imagem")
    plt.ylabel("Probabilidade")
    plt.title("Top 1 (Client X Servidor)")
    plt.legend()
    plt.savefig("diff_probs.png")
    plt.close()


def plot_hist_pred(m_pred):
    plt.figure(figsize=(20, 10))
    plt.hist(m_pred)

    plt.xlabel("Classe", fontsize=20)
    plt.xticks(rotation=90, fontsize="24")
    plt.ylabel("Frequência", fontsize=20)
    plt.title("Frequência das classes preditas", fontsize="24")

    plt.legend()
    plt.savefig("freq_labels_pred.png", bbox_inches="tight")
    plt.close()


def generate_report(
    num_imgs,
    num_server,
    time,
    acc,
    client_probs,
    server_probs,
    inference_time,
    total_time,
    top5,
    total_size_bytes,
):
    report = open("relatorio.md", "w")
    report.write("### Relatório \n")
    report.write(
        "Foram realizadas inferências em "
        + str(num_imgs)
        + " imagens. "
        + "O cliente consultou o servidor em "
        + str(num_server)
        + " predições. "
        + "O tempo médio de inferência por imagem foi de "
        + str(round(time, 4))
        + "s e a acurácia obtida no TOP 1 foi "
        + str(acc)
        + ", enquanto no TOP 5 foi de: "
        + str(top5)
        + "."
        + "O tempo de transferência gasto foi de "
        + str(round(inference_time, 4))
        + "s. O tempo total de execução foi de "
        + str(round(total_time, 4))
        + ".\n"
    )
    report.write("Total de bytes transmitidos: " + str(total_size_bytes))
    plot_diff_probs(client_probs, server_probs)
    report.write("![](diff_probs.png)")


def main(mode, path_root, noisy):
    init_total_time = time.time()
    if mode not in ["server", "client", "hybrid"]:
        print("Invalid mode!")
        return

    base_model = instantiate_model()
    asked_server = False
    url = "https://raw.githubusercontent.com/Mainara/Datasets/master/imagenet1000_labels10.txt"
    f = requests.get(url)
    labels = eval(f.text)

    list_labels = os.listdir(path_root)
    list_labels = random.sample(list_labels, 10)
    pred, prob = None, None
    total_time, transmission_time, top1, count, num_server, top5, total_size_bytes = (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    client_probs, server_probs = [], []
    m_pred = []
    for label in list_labels:
        path_images = path_root + "/" + label
        list_images = os.listdir(path_images)
        for img in list_images:
            image = load_image(path_images + "/" + img, noisy)

            if mode != "server":
                initial_time_client = time.time()
                with torch.no_grad():
                    probs = F.softmax(base_model.forward(image))
                end_time_client = time.time() - initial_time_client
                top_prob, top_pred = torch.topk(probs, 5)
                top_prob, top_pred = top_prob.tolist(), top_pred.tolist()[0]
                pred, prob = int(top_pred[0]), top_prob[0][0]
            if mode == "server" or (
                mode == "hybrid"
                and (
                    top_prob[0][0] >= LOWER_THRESHOLD
                    and top_prob[0][0] <= UPPER_THRESHOLD
                )
            ):
                client_probs.append(prob)
                asked_server = True
                initial_time_server = time.time()
                server_response, size_bytes = server_request(
                    path_images + "/" + img, labels
                )
                total_size_bytes += size_bytes
                top_prob, top_pred = server_response.probs, server_response.preds
                pred, prob = server_response.preds[0], server_response.probs[0]
                end_time_server = time.time() - initial_time_server
                server_probs.append(prob)
                num_server += 1

            count += 1

            print(count)
            labels_5 = [labels[x][1] for x in top_pred]
            if label in labels_5:
                top5 += 1
            if labels[pred][1] == label:
                top1 += 1

            if mode == "client" or (mode == "hybrid" and not asked_server):
                total_time += end_time_client
            elif mode == "server":
                total_time += end_time_server
                transmission_time += end_time_server - server_response.time
            elif mode == "hybrid" and asked_server:
                total_time += end_time_client + end_time_client
                transmission_time += end_time_server - server_response.time
                asked_server = False

    end_total_time = time.time() - init_total_time
    generate_report(
        count,
        num_server,
        (total_time / count),
        (top1 / count),
        client_probs,
        server_probs,
        transmission_time,
        end_total_time,
        (top5 / count),
        total_size_bytes,
    )


try:
    # Establish connection with cloud device
    transport = TSocket.TSocket(SERVER_IP, SERVER_PORT)
    transport = TTransport.TBufferedTransport(transport)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    client = Example.Client(protocol)
    transport.open()

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="path to the image file")
    parser.add_argument("-n", "--noisy", help="put noisy in the image", default=False)
    parser.add_argument("-m", "--mode", help="mode of execution", default="hybrid")
    args = vars(parser.parse_args())

    main(args["mode"], args["image"], args["noisy"])

    transport.close()

except Thrift.TException as tx:
    print(tx.message)
