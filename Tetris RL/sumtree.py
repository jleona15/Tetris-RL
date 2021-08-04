import random

class sumtree:
    def __init__(self, max_depth = 12):
        self.tree = np.zeros((2**max_depth) - 1, dtype = float)
        self.arr = np.ndarray(2**max_depth, dtype = MemoryFragment)
        self.next_insert_index = 0
        self.max_depth = max_depth

    def SUM_CHECK(self):
        depth = 0
        index = 0
        while index * 2 + 2 < self.tree.shape[0]:
            if index + 1 == 2**(depth + 1):
                depth += 1
            if abs(self.tree[index] - (self.tree[index * 2 + 1] + self.tree[index * 2 + 2])) > .00001:
                print("Sum Check error at tree index ", index, ", depth ", depth)
                print(self.tree[index], " != ", self.tree[index * 2 + 1], ' + ', self.tree[index * 2 + 2], ' = ',  self.tree[index * 2 + 1] + self.tree[index * 2 + 2])
                1 / 0

            index += 1

        depth += 1

        while index < self.tree.shape[0]:
            sum_l = 0.
            sum_r = 0.

            if not self.arr[(index * 2 + 1) - self.tree.shape[0]] is None:
                sum_l = self.arr[(index * 2 + 1) - self.tree.shape[0]].loss()

            if not self.arr[(index * 2 + 2) - self.tree.shape[0]] is None:
                sum_r = self.arr[(index * 2 + 2) - self.tree.shape[0]].loss()

            if abs(self.tree[index] - (sum_l + sum_r)) > .00001:
                print("Sum check error at tree index ", index, ", depth ", depth)
                print(self.tree[index], " != ", sum_l, ' + ', sum_r, ' = ',  sum_l + sum_r)
                1 / 0

            index += 1

    def AVG_LOSS(self, i = 0):
        if i * 2 + 2 < self.tree.shape[0]:
            return self.AVG_LOSS(i * 2 + 1) + self.AVG_LOSS(i * 2 + 2)

        l_i = i * 2 + 1 - self.tree.shape[0]
        r_i = i * 2 + 2 - self.tree.shape[0]

        avg = 0.

        if not self.arr[l_i] is None:
            avg += self.arr[l_i].loss()

        if not self.arr[r_i] is None:
            avg += self.arr[r_i].loss()

        return avg / 2
        

    def sample(self, batch_size):
        s = []
        #self.SUM_CHECK();
        for i in range(batch_size):
            s.append(random.random() * self.tree[0])

        return self.sampleHelper(s, 0)

    def sampleHelper(self, s, i):
        if s == []:
            return []
        if i * 2 + 2 < self.tree.shape[0]:

            left_s = []
            right_s = []

            for j in s:
                if self.tree[i * 2 + 1] >= j:
                    left_s.append(j)
                else:
                    right_s.append(j - self.tree[i * 2 + 1])

            #print("##############################")
            #print(s)
            #print(self.tree[i * 2 + 1])
            #print(left_s)
            #print(right_s)
            #print("##############################")

            #print("Left sum: ", self.left.sum)
            #print("Right sum: ", self.right.sum)

            #print("Left samples: ", left_s)
            #print("Right samples: ", right_s)

            return self.sampleHelper(left_s, i * 2 + 1) + self.sampleHelper(right_s, i * 2 + 2)
        
        #print("Parent index: ", (i - 1) // 2)
        #print(self.tree[(i - 1) // 2])

        ret = []
        l_i = i * 2 + 1 - self.tree.shape[0]
        r_i = i * 2 + 2 - self.tree.shape[0]

        #print(i)
        #print(self.arr[0])

        for j in s:
            if self.arr[l_i].loss() >= j:
                ret.append(self.arr[l_i])
            else:
                ret.append(self.arr[r_i])

        return ret

    def insert(self, fragment):
        if self.next_insert_index < self.arr.shape[0]:
            self.arr[self.next_insert_index] = fragment;
            index = self.next_insert_index
            self.next_insert_index += 1
            loss_delta = fragment.loss()
            if self.next_insert_index >= self.arr.shape[0]:
                print("Full at capacity ", self.next_insert_index)
        else:
            
            index = 0
            sample = random.random() * self.tree[0]

            while index * 2 + 2 < self.tree.shape[0]:
                if self.tree[index * 2 + 1] < sample:
                    index = index * 2 + 1
                else:
                    sample -= self.tree[index * 2 + 1]
                    index = index * 2 + 2

            if self.arr[index * 2 + 1 - self.tree.shape[0]].loss() < sample:
                loss_delta = fragment.loss() - self.arr[index * 2 + 1 - self.tree.shape[0]].loss()
                self.arr[index * 2 + 1 - self.tree.shape[0]] = fragment
            else:
                loss_delta = fragment.loss() - self.arr[index * 2 + 2 - self.tree.shape[0]].loss()
                self.arr[index * 2 + 2 - self.tree.shape[0]] = fragment

        index = ((index + self.tree.shape[0]) - 1) // 2

        while index >= 0:
            self.tree[index] += loss_delta
            index = (index - 1) // 2

        #self.SUM_CHECK()

    def updateLogits(self, model):
        for i in range(self.arr.shape[0]):
            if not self.arr[i] is None:
                self.arr[i].updateLogits(model)

        for i in range(2**(self.max_depth - 1) - 1, self.tree.shape[0]):
            new_sum = 0.

            if not self.arr[(i * 2 + 1) - (self.tree.shape[0])] is None:
                new_sum += self.arr[(i * 2 + 1) - (self.tree.shape[0])].loss()

            if not self.arr[(i * 2 + 2) - (self.tree.shape[0])] is None:
                new_sum += self.arr[(i * 2 + 2) - (self.tree.shape[0])].loss()

            self.tree[i] = new_sum

        for i in range(2**(self.max_depth - 1) - 2, -1, -1):
            self.tree[i] = self.tree[i * 2 + 1] + self.tree[i * 2 + 2]



    def updateTargetLogits(self, target_model):
        for i in range(self.arr.shape[0]):
            if not self.arr[i] is None:
                self.arr[i].updateTargetLogits(model)

        for i in range(2**(self.max_depth - 1) - 1, self.tree.shape[0]):
            new_sum = 0.

            if not self.arr[(i * 2 + 1) - (self.tree.shape[0])] is None:
                new_sum += self.arr[(i * 2 + 1) - (self.tree.shape[0])].loss()

            if not self.arr[(i * 2 + 2) - (self.tree.shape[0])] is None:
                new_sum += self.arr[(i * 2 + 2) - (self.tree.shape[0])].loss()

            self.tree[i] = new_sum

        for i in range(2**(self.max_depth - 1) - 2, -1, -1):
            self.tree[i] = self.tree[i * 2 + 1] + self.tree[i * 2 + 2]