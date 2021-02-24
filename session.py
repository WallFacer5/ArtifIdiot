def simple_run(pool, direction):
    def forward(f_pool):
        node_to_run = f_pool[0]
        f_pool = f_pool[1:]
        node_to_run.forward()
        f_pool.extend(list(filter(lambda ol: ol.can_forward, node_to_run.get_output_layers())))
        return f_pool

    def backward(b_pool):
        node_to_run = b_pool[0]
        b_pool = b_pool[1:]
        node_to_run.backward()
        b_pool.extend(list(filter(lambda il: il.can_backward, node_to_run.get_input_layers())))
        return b_pool
    return forward(pool) if direction > 0 else backward(pool)


class Session:
    def __init__(self, ends, x, y, run_algo=simple_run):
        self.ends = ends
        self.starts = set()
        list(map(lambda e: self.starts.update(e.get_starts()), ends))
        self.pool = []
        self.run_algo = run_algo
        self.x, self.y = x, y
        self.cur_pred = []

    def get_pool(self):
        return self.pool

    def run_batch(self, batch_x, batch_y, need_backward=True):
        self.pool = list(self.starts)
        list(map(lambda il: il.set_cur_input(batch_x), self.starts))
        list(map(lambda ol: ol.set_cur_y_true(batch_y), self.ends))
        while self.pool:
            self.pool = self.run_algo(self.pool, direction=1)
        self.cur_pred.extend(self.ends[0].get_cur_outputs())
        if need_backward:
            self.pool = self.ends
            while self.pool:
                self.pool = self.run_algo(self.pool, direction=-1)

    def train_epoch(self, batch_size=1, need_backward=True):
        count = len(self.x)
        self.cur_pred = []
        i = 0
        while i < count:
            self.run_batch(self.x[i:i+batch_size], self.y[i:i+batch_size], need_backward)
            i += batch_size
        # print(self.cur_pred)
        # self.run_batch(self.x, self.y, need_backward=False)
        loss, _ = self.ends[0].loss_function(self.y, self.cur_pred, 0)
        # print(self.y, self.cur_pred)
        return loss

    def train(self, epochs):
        for i in range(epochs):
            loss = self.train_epoch()
            if i % 100 == 0:
                print('epoch: {}; loss: {}.'.format(i, loss))
        print('epoch: {}; loss: {}.'.format(i, loss))
