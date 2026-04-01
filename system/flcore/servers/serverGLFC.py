# system/flcore/servers/serverGLFC.py
import copy
import torch
from flcore.servers.serverbase import Server
from flcore.clients.clientGLFC import clientGLFC

class GLFCServer(Server):
    """
    GLFC server:
    - Keeps a pool of previous global models (proxy teacher); here we use the last best global as teacher.
    - Aggregates client prototypes (mean of normalized features) to broadcast if needed.
    """
    def __init__(self, args, times):
        super().__init__(args, times)
        # Select slow clients if your base provides this API
        try:
            self.set_slow_clients()
        except Exception:
            pass

        # Create clients
        self.set_clients(clientGLFC)

        self.best_global = None
        self.global_prototypes = None  # [C, D]

    def train(self):
        for round in range(self.global_rounds):
            # sample a subset
            self.selected_clients = self.select_clients()
            # broadcast current global model; also share teacher (prev best)
            teacher = self.best_global
            for c in self.selected_clients:
                try:
                    c.set_teacher(teacher)
                except Exception:
                    pass
                self.send_models(c)

            # local updates
            self.receive_models()

            # aggregate as FedAvg (Server base usually has this)
            self.aggregate_parameters()

            # track best global by validation if base supports it
            if hasattr(self, "global_test"):
                acc = self.global_test()
            else:
                acc = None
            if acc is None or (self.rs_test_acc and self.rs_test_acc[-1] >= max(self.rs_test_acc)):
                self.best_global = copy.deepcopy(self.global_model)

            # collect and average client prototypes
            self._aggregate_prototypes()

            # record / display as usual
            self.print_result(round)

        self.save_results()

    def _aggregate_prototypes(self):
        # Gather available client prototypes and average
        protos = []
        for c in self.clients:
            p = getattr(c, "upload_prototypes", None)
            if p is not None:
                protos.append(p)
        if protos:
            P = torch.stack(protos, dim=0).float()
            self.global_prototypes = torch.nanmean(P, dim=0)  # [C, D]
