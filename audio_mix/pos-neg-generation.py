#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import random
import string
from copy import deepcopy as dc
import pandas as pd
import yaml
from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser(description='A simple command-line program.')
parser.add_argument('--audio', '-a', default="./data/final_audio.csv")
parser.add_argument('--sound_class', '-s', default="./data/global_sound.yml")
args = parser.parse_args()

class Node:
    def __init__(self, value=None, left=None, right=None, next=None):
        self.value = value
        self.left = left
        self.right = right
        self.next = next

class Stack:
    def __init__(self):
        self.head = None

    def push(self, node):
        if not self.head:
            self.head = node
        else:
            node.next = self.head
            self.head = node

    def pop(self):
        if self.head:
            popped = self.head
            self.head = self.head.next
            return popped
        else:
            raise Exception("Stack is empty")

class ExpressionTree:

    def get_subexpressions(self, root):
        # print("function enter")
        if root is None:
            return []

        if root.left is None and root.right is None:
            return [root.value]

        subexpressions = []
        left_subs = self.get_subexpressions(root.left)

        right_subs = self.get_subexpressions(root.right)
        for left_sub in left_subs:
            for right_sub in right_subs:
                subexpressions.extend([left_sub, right_sub, '('+left_sub+root.value+right_sub+')'])
        return subexpressions

class Conversion:
    # Constructor to initialize the class variables
    def __init__(self, capacity):
        self.top = -1
        self.capacity = capacity

        # This array is used a stack
        self.array = []

        # Precedence setting
        self.output = []
        self.precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}

    # Check if the stack is empty
    def isEmpty(self):
        return True if self.top == -1 else False

    # Return the value of the top of the stack
    def peek(self):
        return self.array[-1]

    # Pop the element from the stack
    def pop(self):
        if not self.isEmpty():
            self.top -= 1
            return self.array.pop()
        else:
            return "$"

    # Push the element to the stack
    def push(self, op):
        self.top += 1
        self.array.append(op)

    # A utility function to check is the given character
    # is operand
    def isOperand(self, ch):
        return ch.isalpha()

    # Check if the precedence of operator is strictly
    # less than top of stack or not
    def notGreater(self, i):
        try:
            a = self.precedence[i]
            b = self.precedence[self.peek()]
            return True if a <= b else False
        except KeyError:
            return False

    # The main function that
    # converts given infix expression
    # to postfix expression
    def infixToPostfix(self, exp):

        # Iterate over the expression for conversion
        for i in exp:

            # If the character is an operand,
            # add it to output
            if self.isOperand(i):
                self.output.append(i)

            # If the character is an '(', push it to stack
            elif i == '(':
                self.push(i)

            # If the scanned character is an ')', pop and
            # output from the stack until and '(' is found
            elif i == ')':
                while((not self.isEmpty()) and
                      self.peek() != '('):
                    a = self.pop()
                    self.output.append(a)
                if (not self.isEmpty() and self.peek() != '('):
                    return -1
                else:
                    self.pop()

            # An operator is encountered
            else:
                while(not self.isEmpty() and self.notGreater(i)):
                    self.output.append(self.pop())
                self.push(i)

        # Pop all the operator from the stack
        while not self.isEmpty():
            self.output.append(self.pop())

        return "".join(self.output)

def labels_To_Expression(label_exp):
    # 'Microwave oven' * ('Drill' + 'Non-motorized land vehicle' + 'Breaking')
    strings = re.findall(r"'(.*?)'", label_exp)
    notation = ['A','B','C','D']
    replacement_map = dict(zip(strings, notation[:len(strings)]))
    reverse_replacement_map = dict(zip(notation[:len(strings)], strings))

    def replace_string(match):
        string = match.group(1)
        return replacement_map.get(string, string)

    tokens = re.sub(r"'(.*?)'", replace_string, label_exp)
    return tokens.replace(' ', ''), reverse_replacement_map

def sub_Expression_to_label(sub_Exp, mapping):
    translation_table = str.maketrans(mapping)
    return sub_Exp.translate(translation_table)

def sub_Expressions_to_labels(sub_Exps, mapping):
    tokens = []
    for sub_Exp in sub_Exps:
        tokens.append(sub_Expression_to_label(sub_Exp,mapping))
    return tokens

def expression_to_text(sub_exps, xx=None, yy=None):
    if xx is None or yy is None or len(xx)==0 or len(yy)==0:
        xx = [' followed by', ' ends with', ' before', ' and later', ' and then']
        yy = [' overlayed by', ' and', ' with', ' accompanied by', ' surrounded by', ' amidst by']
    sub_exps_text = []
    for sub_exp in sub_exps:
        temp = ''
        for i in sub_exp:
            if i in '()':
                continue
            elif i=='+':
                temp += random.choice(xx)
            elif i=='*':
                temp += random.choice(yy)
            else:
                temp += ' ' + i
        sub_exps_text.append(temp.strip())
    return sub_exps_text

def all_possible_texts(exp):
    all_possible = []
    for xx in [' followed by', ' ends with', ' before', ' and later', ' and then', " proceeded by", " succeeded by", ]:
        for yy in [' overlayed by', ' and', ' with', ' accompanied by', ' surrounded by', ' amidst by', " coupled with" ]:
            temp = ''
            for i in exp:
                if i in '()':
                    continue
                elif i=='+':
                    temp += xx
                elif i=='*':
                    temp += yy
                else:
                    temp += ' ' + i
            all_possible.append([temp.strip(),xx,yy])
    # return random.sample(all_possible, 25)
    return all_possible

def reverse_it(word, ind):

    if word[ind+1] not in '()':
        right_string = word[ind+1]
    else:
        right_string = '('
        count = 1
        temp = ind+2
        while(count):
            right_string+=word[temp]
            if word[temp] not in '()':
                temp+=1
                continue
            if word[temp]=='(':
                count+=1
            else:
                count-=1
            temp+=1
    if word[ind-1] not in '()':
        left_string = word[ind-1]
    else:
        left_string = ')'
        count = 1
        temp = ind-2
        while(count):
            left_string=word[temp] + left_string
            if word[temp] not in '()':
                temp-=1
                continue
            if word[temp]==')':
                count+=1
            else:
                count-=1
            temp-=1

    res1 = ''.join(random.choices(string.digits, k=10))
    res2 = ''.join(random.choices(string.digits, k=10))
    word = word.replace(left_string, res1)
    word = word.replace(right_string, res2)
    word = word.replace(res2, left_string)
    word = word.replace(res1, right_string)
    # print(word)
    return word

def findall(a, b):
    save = []
    for xxx in range(len(a)):
        if a[xxx]==b:
            save.append(xxx)
    return save


def make_negatives(exps):
    all_negs = []
    for i in range(len(exps)):
        findall_temp1 = findall(exps[i], '+')
        findall_temp2 = findall(exps[i], '*')

        for j in findall_temp1:
            temp = dc(exps[i])
            temp = temp[:j] + '*' + temp[j+1:]
            # temp[j] = '*'
            all_negs.append(temp)

        for j in findall_temp2:
            temp = dc(exps[i])
            temp = temp[:j] + '+' + temp[j+1:]
            all_negs.append(temp)


        if '+' in exps[i]:
            ind = []
            for xx in range(len(exps[i])):
                if exps[i][xx]=='+':
                    ind.append(xx)
            for j in ind:
                all_negs.append(reverse_it(exps[i],j))

    return list(set(all_negs))


# In[ ]:



a = pd.read_csv(args.audio)


# In[ ]:


all_exp = list(set(a['sub_exp'].values))


# In[ ]:


with open(args.sound_class, 'r') as file:
    dict_temp = yaml.safe_load(file)


# In[ ]:


all_labels = []
for i in dict_temp.keys():
    all_labels += dict_temp[i]


# In[ ]:


all_labels.sort(key=len, reverse=True)


# In[ ]:


for i in range(len(all_exp)):
    temp = {}
    for j in all_labels:
        if j in all_exp[i]:
            res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            temp[res] = j
            all_exp[i] = all_exp[i].replace(j, res)

    for j in temp.keys():
        all_exp[i] = all_exp[i].replace(j, "'"+temp[j]+"'")


# In[ ]:


final = {}
for i in all_exp:
    # print('expression: ', i)
    s, reverse_replacement_map = labels_To_Expression(i)
    obj = Conversion(len(s))
    s = obj.infixToPostfix(s)

    stack = Stack()
    for c in s:
        if c in "+-*/^":
            z = Node(c)
            x = stack.pop()
            y = stack.pop()
            z.left = y
            z.right = x
            stack.push(z)
        else:
            stack.push(Node(c))
    tree = ExpressionTree()
    sub_exps = list(set(tree.get_subexpressions(stack.pop())))
    sub_exps.sort(key=len, reverse=True)

    caption = sub_exps[0]
    if 'A' not in caption or 'B' not in caption:
        continue
    positives = sub_exps[1:]
    if caption=='(A*B)':
        negatives = ['B+A']
        positives.append('B*A')
    else:
        negatives = make_negatives(sub_exps)
        negatives = [nn for nn in negatives if nn is not caption]

    if len(caption)>5:
        positives = [xx for xx in positives if len(xx)!=1]

    # print('Caption: ', caption)
    # print('Positives: ', positives)
    # print('Negatives: ',negatives)
    # print()

    final[i] = {}
    final[i]['captions'] = {}

    for c in all_possible_texts(caption):
        new_cap = sub_Expression_to_label(c[0], reverse_replacement_map)
        final[i]['captions'][new_cap] = {}
        final[i]['captions'][new_cap]['positives'] = sub_Expressions_to_labels(expression_to_text(positives, [c[1]],[c[2]]),reverse_replacement_map)
        final[i]['captions'][new_cap]['negatives'] = sub_Expressions_to_labels(expression_to_text(negatives, [c[1]],[c[2]]),reverse_replacement_map)

    # print(json.dumps(final,sort_keys=True, indent=4))



# In[ ]:


with open('final.json', 'w') as outfile:
    json.dump(final, outfile)


# In[ ]:


saving = []
new_exps = list(set(a['sub_exp'].values))
for xx in tqdm(range(len(new_exps))):
    corresponding = all_exp[xx]
    if corresponding in final.keys():
        new_a = a[a['sub_exp']==new_exps[xx]].values
        test = random.sample(list(range(len(new_a))), min(len(final[corresponding]['captions'].keys()), len(new_a)))
        test = new_a[test]
        for yy in range(len(test)):
            caption = list(final[corresponding]['captions'].keys())[yy]
            positives = final[corresponding]['captions'][caption]['positives'][:5]
            negatives = final[corresponding]['captions'][caption]['negatives'][:5]
            t = ';'.join([caption] + positives + negatives)
            saving.append([test[yy][0], t, len(positives), len(negatives), "synthetic", "train"])


# In[ ]:


saving_df = pd.DataFrame(saving, columns=['path', 'caption', 'num_of_pos', 'num_of_neg', 'dataset', 'split_name'])
saving_df.to_csv('final_new.csv', index=False)
