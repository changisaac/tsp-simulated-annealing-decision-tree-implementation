"""
CS 486 Assignment 2 Question 3
Author: Isaac Chang 
Date: Nov 3, 2019
"""

import math

def main():
    dt = DecisionTree('../data/horse', 16)
    
    train_file = dt.data_dir + '/horseTrain.txt'
    test_file = dt.data_dir + '/horseTest.txt'

    train_ds = dt.read_in(train_file)
    test_ds = dt.read_in(test_file)

    dt_root = dt.dtl(train_ds, dt.calc_mode(train_ds))

    num_true = 0
    num_false = 0
    correct = 0.0

    for ex in test_ds:
        res = dt.run_dt(dt_root, ex)

        if res:
            num_true += 1
        elif not res:
            num_false += 1

        if res == ex[dt.class_idx]:
            correct += 1

    accuracy = correct / len(test_ds)

    print 'Accuracy: ' + str(accuracy*100) + '%'
    print 'Number Healthy: ' + str(num_true)
    print 'Number Colic: ' + str(num_false)

    print dt.print_dt(dt_root,0)

class DecisionTree:
    
    def __init__(self, data_dir, class_idx):
        self.data_dir = data_dir
        self.class_idx = class_idx

    def run_dt(self, node, ex):
        if node.class_val is not None:
            return node.class_val
        elif ex[node.ftr] < node.thresh:
            return self.run_dt(node.left, ex)
        elif ex[node.ftr] >= node.thresh:
            return self.run_dt(node.right, ex)

    def print_ds(self, ds):
        for ex in ds:
            print ex
   
    # for horse data set
    def print_dt(self, dt, depth):
        ftr_names = ['K','Na','CL','HCO3','Endotoxin','Aniongap','PLA2','SDH','GLDH','TPP',
                     'Breath rate', 'PCV', 'Pulse rate','Fibrinogen','Dimer','FibPerDim']
        
        out = ''

        if dt.right != None:
            out += self.print_dt(dt.right, depth+1)

        if dt.ftr != None:
            out += "\n" + ("  "*depth) + "{}, {}".format(ftr_names[dt.ftr], dt.thresh)
        else:
            out += "\n" + ("  "*depth) + str(dt.class_val)

        if dt.left != None:
            out += self.print_dt(dt.left, depth + 1)

        return out

    def dtl(self, ds, default):
        # no examples
        if len(ds) == 0:
            return self.Node(class_val=default)
        # all examples give the same classification
        elif self.check_same_class(ds)[0]:
            return self.Node(class_val=self.check_same_class(ds)[1])
        # no features only classifications
        elif len(ds[0]) == 0:
            raise
            return self.Node(class_val=self.calc_mode(ds))
        else:
            # choose feature and thresh for next node
            ftr_num, thresh, info_gain = self.choose_ftr(ds)
            root = self.Node(ftr_num, info_gain, thresh)

            # get subsets of data for dtl to recurse on
            # subset of data with ftr_num above thresh
            above_ds = self.get_sub_ds(ds, ftr_num, thresh, True)
            # subset of data with ftr_num below thresh
            below_ds = self.get_sub_ds(ds, ftr_num, thresh, False)
           
            # set to left and right branch of tree
            root.right = self.dtl(above_ds, self.calc_mode(above_ds))
            root.left = self.dtl(below_ds, self.calc_mode(below_ds))

            return root

    # finds the feature to use at node in decision tree based on max yielded info gain
    def choose_ftr(self, ds):
        max_info_gain = float('-inf')
        ret_ftr = None
        ret_thresh = None
        ret_info_gain = None
        
        # loop through all features index 0 to last_idx-1
        for i in range(len(ds[0])-1):
            # sort examples in data set by the feature
            ds.sort(key=lambda x: x[i])
            # find all possible thresholds
            threshs = self.calc_threshs(ds, i)
           
            # loop through all possible threshs for each feature
            # and for each thresh calculate the resulting info gain 
            for thresh in threshs:
                info_gain = self.calc_info_gain(ds, i, thresh)
                
                # keep track of largest info gain case 
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    ret_info_gain = info_gain
                    ret_ftr = i
                    ret_thresh = thresh

        return (ret_ftr, ret_thresh, ret_info_gain)
            
    def calc_mode(self, ds):
        p, n = self.calc_pos_neg(ds)

        if p >= n:
            return True
        else:
            return False

    def check_same_class(self, ds):
        p, n = self.calc_pos_neg(ds)

        if p == 0 or n == 0:
            if p == 0:
                return (True, False)
            elif n == 0:
                return (True, True)

        return (False, None)

    # cycle through all examples in data set to calculate all possible thresholds
    # based on halfway value between successive features values
    def calc_threshs(self, ds, ftr_num):
        threshs = []

        for i in range(1, len(ds)):
            curr = ds[i][ftr_num]
            prev = ds[i-1][ftr_num] 
            
            if curr != prev:
                thresh = (curr + prev) / 2
                threshs.append(thresh)

        return threshs

    def calc_info_gain(self, ds, ftr_num, thresh):
        entr = self.calc_entr(ds)
        rem = self.calc_rem(ds, ftr_num, thresh)

        info_gain = entr - rem

        return info_gain

    def calc_rem(self, ds, ftr_num, thresh):
        # first need to split data set into its 2 states, above and below thresh
        above_ds = self.get_sub_ds(ds, ftr_num, thresh, True)
        below_ds = self.get_sub_ds(ds, ftr_num, thresh, False)

        # calc entropy for either subset
        above_entr = self.calc_entr(above_ds)
        below_entr = self.calc_entr(below_ds)

        total = float(len(ds))

        rem = ((len(above_ds) / total) * above_entr) + ((len(below_ds) / total) * below_entr)

        return rem

    def calc_entr(self, ds):
        if len(ds) == 0:
            return 0.0

        p, n = self.calc_pos_neg(ds)

        total = p + n

        p_prob = p / total
        n_prob = n / total

        entr = None

        if p_prob == 0:
            entr = -(n_prob * math.log(n_prob, 2.0)) 
        elif n_prob == 0:
            entr = -(p_prob * math.log(p_prob, 2.0))
        else:
            entr = -(p_prob * math.log(p_prob, 2.0)) - (n_prob * math.log(n_prob, 2.0))

        return entr

    def get_sub_ds(self, ds, ftr_num, thresh, above_thresh):
        sub_ds = []

        for i in range(len(ds)):
            if above_thresh:
                if ds[i][ftr_num] >= thresh:
                    sub_ds.append(ds[i][:])
            elif not above_thresh: 
                if ds[i][ftr_num] < thresh:
                    sub_ds.append(ds[i][:])

        return sub_ds

    def calc_pos_neg(self, ds):
        p = 0.0
        n = 0.0
       
        for i in range(len(ds)):
            if ds[i][self.class_idx]:
                p+= 1
            else:
                n+= 1

        return (p, n)
    
    # Helper Classes -----
    
    # Struct class representing a node in the decision tree
    class Node:
        
        def __init__(self, ftr=None, info_gain=None, thresh=None, left=None, right=None, class_val=None):
            # to represent leaf node with classified result (true, false)
            # if class_val is used all other terms are None and not used
            self.class_val = class_val
            self.ftr = ftr
            self.info_gain = info_gain
            self.thresh = thresh
            self.left = left
            self.right = right

        def __str__(self):
            return "Feature: {}, Threshold: {}, Info Gain: {}, Classification: {}".format(self.ftr, self.thresh, self.info_gain, self.class_val)

    # Setup Functions -----

    def read_in(self, file_name):
        f = open(file_name)
        lines = list(f)

        ftrs = []

        for i in range(len(lines)):
            ftr = lines[i].split(',')
            ftr_float = [float(num) for num in ftr[:len(ftr)-1]]
            
            if 'healthy' in ftr[-1]:
                ftr_float.append(True)
            else:
                ftr_float.append(False)

            ftrs.append(ftr_float)

        # data returned as ds[example #][feature #]
        return ftrs

if __name__ == '__main__':
    main()
