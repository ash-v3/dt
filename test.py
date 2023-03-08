import torch

name = "saved_buffer-09-10-31.pt"
replay_buffer = torch.load(name)

# print(buffer.buffer)

model = torch.load("model.pt")

# epoch = replay_buffer.buffer[0]
# input = epoch[:-1]
# res = model(input)
# print(input.shape, res.shape, replay_buffer.buffer[0].shape)

# print(res - replay_buffer.buffer[0][-1])

input = torch.randn([2, 772])
res0 = model(input)
res1 = model(input)

print(res0 - res1)
