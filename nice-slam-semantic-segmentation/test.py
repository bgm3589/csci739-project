import torch
from src.common import quad2rotation, get_tensor_from_camera, get_camera_from_tensor
device = "cuda:0"
camera = torch.tensor( [[-0.947387, 0.126025 , -0.294239 ,2.649368], [0.319939 , 0.401221 ,-0.858289 ,2.978560],
[0.009889 ,-0.907270 ,-0.420431 ,1.365403],
[0.000000 ,0.000000 ,0.000000 ,1.000000]] , device=device)
quad = get_tensor_from_camera(camera).to(device)
cmr = get_camera_from_tensor(quad)
print(camera)
print(quad)
print(cmr)