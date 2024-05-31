"""This script simulates a knapsack problem where objects of different sizes are processed and placed into a box with a
limited capacity. The script uses a max heap to keep track of the objects being processed and places them into the box

This is for brainstorming a solution to the dynamic memory optimization problem
"""
import time
from typing import Any


class KeyedObject:
    def __init__(self, key: Any):
        self.key = key


class AVLNode(KeyedObject):
    def __init__(self, key: Any):
        super().__init__(key)
        self.left = None
        self.right = None
        self.parent = None
        self.height = 1
        self.min = key
        self.name = None  # should be a process object


class AVLTree():
    def __init__(self, arr: list[KeyedObject] = None):
        self.root = None
        if arr:
            self.buildAVL(arr)

    def buildAVL(self, arr: list[KeyedObject]):
        for obj in arr:
            self.insert(obj)

    def insert(self, obj: KeyedObject):
        """inserts a node in an AVLTree"""
        self.root = self._insert(self.root, obj)

    def _insert(self, root: AVLNode, obj: KeyedObject):
        if not root:
            return AVLNode(obj.key)
        if obj.key < root.key:
            root.left = self._insert(root.left, obj)
        else:
            root.right = self._insert(root.right, obj)
        root.height = 1 + max(self.getHeight(root.left), self.getHeight(root.right))
        balance = self.getBalance(root)
        if balance > 1:
            if obj.key < root.left.key:
                return self.rightRotate(root)
            else:
                root.left = self.leftRotate(root.left)
                return self.rightRotate(root)
        if balance < -1:
            if obj.key > root.right.key:
                return self.leftRotate(root)
            else:
                root.right = self.rightRotate(root.right)
                return self.leftRotate(root)

        root.min = min(root.key, root.left.min, root.right.min)
        return root

    def delete(self, obj: AVLNode | KeyedObject):
        """deletes a node in an AVLTree"""
        if not isinstance(obj, AVLNode):
            obj = self.find(obj.key, by_data=True)
        self.root = self._delete(self.root, obj)

    def _delete(self, root: AVLNode, obj: AVLNode):
        if not root:
            return root
        if obj.key < root.key:
            root.left = self._delete(root.left, obj)
        elif obj.key > root.key:
            root.right = self._delete(root.right, obj)
        else:
            if not root.left or not root.right:
                root = root.left or root.right
            else:
                temp = self.getMin(root.right)
                root.key = temp.key
                root.right = self._delete(root.right, temp)
        if not root:
            return root
        root.height = 1 + max(self.getHeight(root.left), self.getHeight(root.right))
        balance = self.getBalance(root)
        if balance > 1:
            if self.getBalance(root.left) >= 0:
                return self.rightRotate(root)
            else:
                root.left = self.leftRotate(root.left)
                return self.rightRotate(root)
        if balance < -1:
            if self.getBalance(root.right) <= 0:
                return self.leftRotate(root)
            else:
                root.right = self.rightRotate(root.right)
                return self.leftRotate(root)
        if root:
            root.min = min(root.key, root.left.min, root.right.min)
        else:
            root.min = root.key
        return root

    def leftRotate(self, z: AVLNode):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self.getHeight(z.left), self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))
        return y

    def rightRotate(self, z: AVLNode):
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        z.height = 1 + max(self.getHeight(z.left), self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))
        return y

    def getHeight(self, root: AVLNode):
        if not root:
            return 0
        return root.height

    def getBalance(self, root: AVLNode):
        if not root:
            return 0
        return self.getHeight(root.left) - self.getHeight(root.right)

    def getMin(self, right):
        if not right.left:
            return right
        return self.getMin(right.left)

    def getMax(self, left):
        if not left.right:
            return left
        return self.getMax(left.right)

    def find(self, key: Any, by_data=False):
        return self._find(self.root, key, by_data)

    def _find(self, root: AVLNode, key: Any, by_data=False):
        if not root:
            return None
        if (root.key == key and not by_data) or (root.name == key and by_data):
            return root
        if (root.key > key and not by_data) or (root.name > key and by_data):
            return self._find(root.left, key, by_data)
        return self._find(root.right, key, by_data)

    def is_empty(self):
        return self.root is None

    def getMax_with_threshold(self, threshold: int, start: AVLNode = None):
        """finds the maximum object with key less than the threshold"""
        if not start:
            start = self.root
        return self._getMax_with_threshold(start, threshold)

    def _getMax_with_threshold(self, start: AVLNode, threshold: int):
        curr = start
        while curr.right and curr.right.key < threshold:
            curr = curr.right
        if curr.key < threshold:
            return curr
        elif curr.left:
            return self._getMax_with_threshold(curr.left, threshold)
        else:
            return None


class MaxHeap():
    def __init__(self, arr: list[KeyedObject] = None):
        """builds a max heap from a list of objects"""
        if arr:
            self.heap = arr
            self.buildMaxHeap()
        else:
            self.heap = []

    def __len__(self):
        return len(self.heap)

    def buildMaxHeap(self):
        """takes O(n) using bottom-up approach"""
        for i in range(len(self) // 2, -1, -1):
            self.heapify(i)

    def heapify(self, i):
        """takes O(log n)"""
        curr = i
        left, right = 2 * i + 1, 2 * i + 2
        if left < len(self) and self.heap[left].key > self.heap[curr].key:
            curr = left
        if right < len(self) and self.heap[right].key > self.heap[curr].key:
            curr = right
        if curr != i:
            self.heap[i], self.heap[curr] = self.heap[curr], self.heap[i]
            self.heapify(curr)

    def extractMax(self):
        """takes O(log n)"""
        if len(self) == 0:
            return None
        max_val = self.heap[0]
        self.heap[0] = self.heap[-1]  # move last elem to root
        self.heap.pop()  # removes last elem of list
        self.heapify(0)  # heapify from root
        return max_val

    def insert(self, val: Any):
        """takes O(log n)"""
        self.heap.append(val)
        i = len(self) - 1
        while i > 0 and self.heap[(i - 1) // 2].key < self.heap[i].key:
            self.heap[(i - 1) // 2], self.heap[i] = self.heap[i], self.heap[(i - 1) // 2]
            i = (i - 1) // 2

    def max(self):
        """returns the maximum value in the heap"""
        return self.heap[0]

    def remove(self, val: Any):
        """removes a value from the heap"""
        for i in range(len(self)):
            if self.heap[i].key == val:
                self.heap[i].key = float('inf')
                self.heapify(i)
                self.extractMax()

    def is_empty(self):
        return len(self) == 0


class Process(KeyedObject):
    def __init__(self, size: int, data: Any):
        super().__init__(size)
        self.size = size
        self.size_remaining = size
        self.name = data
        self.running = False
        self.completed = False

    def start(self):
        try:
            if self.completed:
                assert False, "Cannot start a completed process"
            self.running = True
        except AssertionError as e:
            print(e)

    def finish(self):
        self.running = False
        self.completed = True
        print(f"Process {self.name} has completed.")


class ProcessManager:
    """Manages the processes and the box where they are placed.
    """
    def __init__(self, datasets: list[Process], total_space=64):
        self.num_procs = len(datasets)
        self.running_procs = set()
        self.pot_proc_heap = MaxHeap(datasets)
        self.pot_proc_avl = AVLTree(datasets)

        self.completed_processes = 0
        self.total_space = total_space  # this is a constant
        self.remaining_space = total_space  # this changes as processes are added and removed

        self.ready_for_new_process = True

    def run(self):
        while self.completed_processes < self.num_procs:
            if self.pot_proc_heap.is_empty() or self.pot_proc_avl.is_empty():  # for safety
                break
            # load new processes onto running set
            if self.ready_for_new_process:
                self.try_start_new_processes()

            while not self.ready_for_new_process:
                self.process_all_procs()

    def process_all_procs(self):
        """simulates 1 second of processing time for all running processes
        The simulation naively assumes 1 size unit is processed per second"""
        for process in self.running_procs:
            if process.size_remaining > 0:
                process.size_remaining -= 1
            else:
                self.finish_process(process)
        time.sleep(1)  # simulate waiting time for processes to be processed

    def try_start_new_processes(self) -> None:
        """Tries to start as many new processes as it can with the available space
        (refer to notes for the algorithm explanation)
        """
        if self.pot_proc_avl.root.min > self.remaining_space:
            print("No process can fit in the remaining space")
            self.ready_for_new_process = False
            return

        process = self.pot_proc_heap.max()  # gets the largest process
        if process.size > self.total_space:
            self.pot_proc_heap.extractMax()
            self.pot_proc_avl.delete(process)
            self.ignore_process(process, "process size exceeds total possible space")

        elif process.size <= self.remaining_space:
            self.pot_proc_heap.extractMax()
            if not (process.running or process.completed):
                # if we got here, it means we're encountering a normal process (i.e. this is NOT a ghost process)
                # as opposed to a 'ghost process' which was run via removing from the avl, but hasn't been removed from the heap yet
                self.pot_proc_avl.delete(process)
                self.start_process(process)

        else:  # max heap's max is too big, but there's still space left, so we'll look for a smaller process in the AVL
            trying = self.pot_proc_avl.getMax_with_threshold(self.remaining_space, None)
            process = trying.data  # note that the avl stores the process object in the data field
            if process:
                # note that we don't remove from heap, to save runtime. But don't worry - it'll be handled
                self.pot_proc_avl.delete(trying)
                self.start_process(trying)
            else:
                print("No process can fit in the remaining space (tried AVL tree)")
                self.ready_for_new_process = False
                return

        self.try_start_new_processes()

    def start_process(self, process):
        process.start()
        self.running_procs.add(process)
        self.remaining_space -= process.size
        print(f"Process {process.name} has started.")

    def finish_process(self, process):
        process.finish()
        self.running_procs.remove(process)
        self.completed_processes += 1
        self.remaining_space += process.size
        print(f"Process {process.name} has finished.")

        self.ready_for_new_process = True

    def ignore_process(self, process, reason=""):
        self.completed_processes += 1
        print(f"Process {process.name} has been ignored" + (f" due to {reason}." if reason else "."))


if __name__ == "__main__":
    objects = [
        Process(6, "Paramyxoviridae-5"),
        Process(8, "Mitoviridae-51"),
        Process(9, "Quenyaviridae-5"),
        Process(5, "Secoviridae-16"),
        Process(2, "Endornaviridae-16"),
        Process(4, "Phenuiviridae-9")
    ]
    proc_man = ProcessManager(objects, 16)
    proc_man.run()
