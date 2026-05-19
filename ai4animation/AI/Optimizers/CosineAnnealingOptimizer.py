from ai4animation.AI.Optimizers.AdamWR.AdamW import AdamW
from ai4animation.AI.Optimizers.AdamWR.CyclicScheduler import CyclicScheduler


class CosineAnnealingOptimizer:
    def __init__(
        self,
        params,
        batch_size,
        batch_count,
        lr=1e-4,
        decay=1e-4,
        restart_period=10,
        t_mult=2,
    ):
        self.Optimizer = AdamW(params, lr=lr, weight_decay=decay)
        self.Scheduler = CyclicScheduler(
            optimizer=self.Optimizer,
            batch_size=batch_size,
            epoch_size=batch_count * batch_size,
            restart_period=restart_period,
            t_mult=t_mult,
            policy="cosine",
            verbose=True,
        )
        self.Step = 0
        self.Total = batch_count

    def Update(self, loss):
        if isinstance(loss, dict):
            loss = sum(loss.values())
        self.Optimizer.zero_grad()
        loss.backward()
        self.Optimizer.step()
        self.Scheduler.batch_step()
        self.Step += 1
        if self.Step == self.Total:
            self.Scheduler.step()
            self.Step = 0
            # print(" Optimizer: Epoch complete.", end="\r")
