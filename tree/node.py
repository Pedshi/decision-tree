class Node:
    def __init__(self, ig, attr_name):
        self.ig = ig
        self.attr_name = attr_name
        self.branchs = list()

    def add_branch(self, branch):
        self.branchs.append(branch)
