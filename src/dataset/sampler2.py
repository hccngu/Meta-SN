import time
import numpy as np

from queue import Queue
import dataset.utils as utils



class SerialSampler():

    def __init__(self, data, args, sampled_classes, source_classes, num_episodes=None, example_prob_metrix=None):
        self.data = data
        self.args = args
        self.num_episodes = num_episodes
        self.sampled_classes = sampled_classes
        self.source_classes = source_classes
        self.example_prob_metrix = example_prob_metrix

        self.all_classes = np.unique(self.data['label'])
        self.num_classes = len(self.all_classes)
        if self.num_classes < self.args.way:
            raise ValueError("Total number of classes is less than #way.")

        self.idx_list = []
        for y in self.all_classes:
            self.idx_list.append(
                np.squeeze(np.argwhere(self.data['label'] == y)))

        self.count = 0
        self.done_queue = Queue()
        self.worker(self.done_queue, self.sampled_classes, self.source_classes)


    def get_epoch(self):

        for _ in range(self.num_episodes):
            # wait until self.thread finishes
            support, query = self.done_queue.get()

            # convert to torch.tensor
            support = utils.to_tensor(support, self.args.cuda, ['raw'])
            query = utils.to_tensor(query, self.args.cuda, ['raw'])

            support['is_support'] = True
            query['is_support'] = False

            yield support, query

    def worker(self, done_queue, sampled_classes, source_classes):
        '''
            Generate one task (support and query).
            Store into self.support[self.cur] and self.query[self.cur]
        '''
        example_prob_metrix = self.example_prob_metrix
        while True:
            if done_queue.qsize() > self.num_episodes:
                time.sleep(1)
                return
                # continue

            # sample examples
            support_idx, query_idx = [], []
            if example_prob_metrix is None:
                for y in sampled_classes:
                    tmp = np.random.permutation(len(self.idx_list[y]))
                    if len(tmp) < self.args.shot + self.args.query:
                        tmp = np.random.choice(len(self.idx_list[y]), self.args.shot + self.args.query, replace=True)

                    support_idx.append(
                        self.idx_list[y][tmp[:self.args.shot]])
                    query_idx.append(
                        self.idx_list[y][tmp[self.args.shot:self.args.shot + self.args.query]])
            else:
                for y in sampled_classes:
                    tmp = np.random.choice(len(self.idx_list[y]), self.args.shot + self.args.query, p=example_prob_metrix[y][0], replace=False)
                    support_idx.append(
                        self.idx_list[y][tmp[:self.args.shot]])
                    query_idx.append(
                        self.idx_list[y][
                            tmp[self.args.shot:]])

            support_idx = np.concatenate(support_idx)
            query_idx = np.concatenate(query_idx)

            # aggregate examples
            max_support_len = np.max(self.data['text_len'][support_idx])
            max_query_len = np.max(self.data['text_len'][query_idx])

            support = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
                                          support_idx, max_support_len)
            query = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
                                        query_idx, max_query_len)

            done_queue.put((support, query))

    def __del__(self):
        '''
            Need to terminate the processes when deleting the object
        '''

        del self.done_queue



def task_sampler(data, args, classes_sample_p=None):
    all_classes = np.unique(data['label'])
    num_classes = len(all_classes)

    _, id_metrix = np.mgrid[0: num_classes: 1, 0: num_classes: 1]  # [N, N]每行都是0~(N-1)
    id_metrix = id_metrix[~np.eye(id_metrix.shape[0], dtype=bool)].reshape(id_metrix.shape[0], -1)  # 去掉了对角线元素

    # sample classes
    if classes_sample_p is None:
        temp = np.random.permutation(num_classes)
        sampled_classes = temp[:args.way]
    else:
        class_names_num = []
        class_name_num = np.random.choice(len(all_classes), 1)
        a = class_name_num[0]
        class_names_num.append(a)
        p = classes_sample_p[a]
        for i in range(args.way - 1):
            class_name_num = np.random.choice(id_metrix[a], 1, p=p, replace=False)
            a = class_name_num[0]
            if a in class_names_num:
                t1 = np.arange(len(all_classes))
                t2 = []
                for k in t1:
                    if k not in class_names_num:
                        t2.append(k)
                # print("t1", t1)
                a = np.random.choice(t2, 1)[0]
            class_names_num.append(a)
            p = (p + classes_sample_p[a]) / 2

        sampled_classes = class_names_num

    source_classes = None
    # print("sampled_classes", sampled_classes)

    return sampled_classes, source_classes
