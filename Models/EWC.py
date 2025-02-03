
import torch



class EWC(torch.nn.Module):
     
   
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(EWC, self).__init__()
        self.reg = args.memory_strength

        # setup network
        self.is_cifar = True  # Explicitly using CIFAR-100 (ResNet18)
        self.net = ResNet18(n_outputs).to(torch.device("cuda:2" if args.cuda else "cpu"))

        # setup optimizer
        self.opt = torch.optim.SGD(self.net.parameters(), lr=args.lr)

        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()

        # setup memories
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.memx = None
        self.memy = None

        self.nc_per_task = n_outputs // n_tasks  # Number of classes per task
        self.n_outputs = n_outputs
        self.n_memories = args.n_memories

        self.device = torch.device("cuda:2" if args.cuda else "cpu")
        self.nt = n_tasks
        self.reg = args.memory_strength
        self.n_memories = args.n_memories
        self.num_exemplars = 0
        self.samples_per_task = args.samples_per_task
        self.examples_seen = 0
        if self.is_cifar:
            self.nc_per_task = n_outputs / n_tasks
        else:
            self.nc_per_task = n_outputs
        self.n_outputs = n_outputs
        self.n_memories = args.n_memories

        # Initialize examples_seen
        self.examples_seen = 0  # Tracks the number of examples processed for the current task

    # Other methods remain unchanged


    def compute_offsets(self, task):
        if self.is_cifar:
            offset1 = task * self.nc_per_task
            offset2 = (task + 1) * self.nc_per_task
        else:
            offset1 = 0
            offset2 = self.n_outputs
        return int(offset1), int(offset2)

    def forward(self, x, t):
        output = self.net(x)
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):
        self.net.train()

        # next task?
        if t != self.current_task:
            self.net.zero_grad()

            if self.is_cifar:
                offset1, offset2 = self.compute_offsets(self.current_task)
                self.bce((self.net(self.memx)[:, offset1: offset2]),
                         self.memy - offset1).backward()
            else:
                self.bce(self(self.memx,
                              self.current_task),
                         self.memy).backward()
            self.fisher[self.current_task] = []
            self.optpar[self.current_task] = []
            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.optpar[self.current_task].append(pd)
                self.fisher[self.current_task].append(pg)
            self.current_task = t
            self.memx = None
            self.memy = None

        if self.memx is None:
            self.memx = x.data.clone()
            self.memy = y.data.clone()
        else:
            if self.memx.size(0) < self.n_memories:
                self.memx = torch.cat((self.memx, x.data.clone()))
                self.memy = torch.cat((self.memy, y.data.clone()))
                if self.memx.size(0) > self.n_memories:
                    self.memx = self.memx[:self.n_memories]
                    self.memy = self.memy[:self.n_memories]

        self.net.zero_grad()
        if self.is_cifar:
            offset1, offset2 = self.compute_offsets(t)
            loss = self.bce((self.net(x)[:, offset1: offset2]),
                            y - offset1)
        else:
            loss = self.bce(self(x, t), y)
        for tt in range(t):
            for i, p in enumerate(self.net.parameters()):
                l = self.reg * self.fisher[tt][i]
                l = l * (p - self.optpar[tt][i]).pow(2)
                loss += l.sum()
        loss.backward()
        self.opt.step()
        # Handle the last minibatch of the current task
        if self.examples_seen == self.samples_per_task:
            self.examples_seen = 0
            all_labs = torch.unique(self.memy)
            num_classes = all_labs.size(0)
            self.num_exemplars = int(self.n_memories / (num_classes + len(self.mem_class_x.keys())))
            offset1, offset2 = self.compute_offsets(t)

            for lab in all_labs:
                lab = lab.item()
                indxs = (self.memy == lab).nonzero(as_tuple=True)[0]
                cdata = self.memx[indxs]

                mean_feature = self.net(cdata).data.mean(0)
                exemplars = torch.zeros(self.num_exemplars, *x.size()[1:], device=self.device)
                taken = torch.zeros(cdata.size(0), device=self.device)
                for ee in range(self.num_exemplars):
                    prev = torch.zeros_like(mean_feature, device=self.device)
                    if ee > 0:
                        prev = self.net(exemplars[:ee]).data.sum(0)
                    cost = (mean_feature.expand(cdata.size(0), -1) - (self.net(cdata).data + prev.expand(cdata.size(0), -1)) / (ee + 1)).norm(2, dim=1)
                    _, indx = cost.sort(0)
                    winner = 0
                    while winner < indx.size(0) and taken[indx[winner]] == 1:
                        winner += 1
                    if winner < indx.size(0):
                        taken[indx[winner]] = 1
                        exemplars[ee] = cdata[indx[winner]].clone()
                    else:
                        exemplars = exemplars[:indx.size(0)].clone()
                        self.num_exemplars = indx.size(0)
                        break

                self.mem_class_x[lab] = exemplars.clone()

            for cc in self.mem_class_x.keys():
                self.mem_class_y[cc] = self.net(self.mem_class_x[cc]).data.clone()

            self.memx = None
            self.memy = None
def train_EWC_on_cifar100(args):
    device = torch.device("cuda:2" if args.cuda else "cpu")
    train_tasks, test_tasks = load_cifar100_tasks(args)
    model = EWC(32 * 32 * 3, 100, args.n_tasks, args).to(device)

    task_accuracies = []
    for task_id, (train_loader, test_loader) in enumerate(zip(train_tasks, test_tasks)):
        model.train()
        for epoch in range(args.n_epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                model.observe(x, task_id, y)

        # Evaluate
        model.eval()
        accuracies = []
        for test_id, test_loader in enumerate(test_tasks[:task_id + 1]):
            correct, total = 0, 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    output = model(x, test_id)
                    _, pred = output.max(1)
                    correct += pred.eq(y).sum().item()
                    total += y.size(0)
            accuracy = 100 * correct / total
            accuracies.append(accuracy)
            print(f"Task {task_id + 1}, Test on Task {test_id + 1}, Accuracy: {accuracy:.2f}%")
        task_accuracies.append(accuracies)

    return task_accuracies

# CIFAR-100 Task Loader
def load_cifar100_tasks(args):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform)
    test_data = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform)

    classes_per_task = 100 // args.n_tasks
    train_tasks, test_tasks = [], []

    for task in range(args.n_tasks):
        task_classes = range(task * classes_per_task, (task + 1) * classes_per_task)
        train_idx = [i for i, target in enumerate(train_data.targets) if target in task_classes]
        test_idx = [i for i, target in enumerate(test_data.targets) if target in task_classes]

        train_tasks.append(DataLoader(Subset(train_data, train_idx), batch_size=args.batch_size, shuffle=True))
        test_tasks.append(DataLoader(Subset(test_data, test_idx), batch_size=args.batch_size, shuffle=False))

    return train_tasks, test_tasks

# Set arguments
class Args:
    data_path = './data'
    n_tasks = 20  # Number of tasks
    n_memories = 200
    memory_strength = 0.5
    samples_per_task = 5000  # Total training samples per task
    n_epochs = 1
    batch_size = 10
    lr = 1e-3
    cuda = torch.cuda.is_available()
    n_layers = 2  # Number of layers for MLP (if used)
    n_hiddens = 100  # Number of hidden units per layer for MLP (if used)
    data_file = 'cifar100.pt'  # Ensure this matches CIFAR-100 dataset usage

# Run training
args = Args()
task_accuracies = train_EWC_on_cifar100(args)

