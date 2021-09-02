import torch

if torch.cuda.is_available():
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    device = torch.device("cuda")
    print("current GPU :", torch.cuda.current_device())
    print('GPU:', torch.cuda.get_device_name)
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
