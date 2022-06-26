#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:39:29 2022

@author: padela
"""


# %% Define an hash-table class
class HashTable:

    # Create empty bucket list of given size
    def __init__(self, size):
        self.size = size
        self.hash_table = self.create_buckets()

    def create_buckets(self):
        return [list() for _ in range(self.size)]

    # Insert values into hash map
    def set_val(self, key, val):

        # Get the index from the key
        # using hash function
        hashed_key = hash(key) % self.size

        # Get the bucket corresponding to index
        bucket = self.hash_table[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record

            # check if the bucket has same key as
            # the key to be inserted
            if record_key == key:
                found_key = True
                break

        # If the bucket has same key as the key to be inserted,
        # Update the key value
        # Otherwise append the new key-value pair to the bucket
        if found_key:
            bucket[index] = (key, val)
        else:
            bucket.append((key, val))

    # Return searched value with specific key
    def get_val(self, key):

        # Get the index from the key using
        # hash function
        hashed_key = hash(key) % self.size

        # Get the bucket corresponding to index
        bucket = self.hash_table[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record

            # check if the bucket has same key as
            # the key being searched
            if record_key == key:
                found_key = True
                break

        # If the bucket has same key as the key being searched,
        # Return the value found
        # Otherwise indicate there was no record found
        if found_key:
            return record_val
        else:
            return "No record found"

    # Remove a value with specific key
    def delete_val(self, key):

        # Get the index from the key using
        # hash function
        hashed_key = hash(key) % self.size

        # Get the bucket corresponding to index
        bucket = self.hash_table[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record

            # check if the bucket has same key as
            # the key to be deleted
            if record_key == key:
                found_key = True
                break
        if found_key:
            bucket.pop(index)
        return

    # To print the items of hash map
    def __str__(self):
        return "".join(str(item) for item in self.hash_table)


def str_list(lst):
    if isinstance(lst, str):
        return lst
    else:
        s = lst[0]
        for i in lst[1:]:
            s += ', ' + str(i)
        return s


def list_str(string=str):
    if isinstance(string, list):
        return string
    else:
        return string.split(', ')


def trace_as_vec():
    raise NotImplementedError()


def frequency_table(X_train, case_id_name=str, activity_name=str, thrs=.2):
    freq_dict = dict()

    idx_list = X_train[case_id_name].unique()

    for trace_idx in idx_list:

        # Get the whole list of activities
        trace = X_train[X_train[case_id_name] == trace_idx].reset_index(drop=True)
        if str_list(trace[activity_name]) in freq_dict.keys():
            freq_dict[str_list(trace[activity_name])] += 1

        else:
            freq_dict[str_list(trace[activity_name])] = 1

    # Order by frequence and get the first
    freq_dict = list({k: v for k, v in sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)}.keys())
    freq_dict = set(freq_dict[:int(len(freq_dict) * thrs + 1)])

    return freq_dict


def fill_hashmap(X_train, case_id_name=str, activity_name=str, thrs=.2):
    if thrs == .2:
        print('Creating hash-map following Pareto\'s principle')
        traces_selected = frequency_table(X_train, case_id_name=case_id_name, activity_name=activity_name, thrs=thrs)

    elif thrs != 0:
        print(f'Creating hash-map with a threshold of {thrs} of acceptability for outliers')
        traces_selected = frequency_table(X_train, case_id_name=case_id_name, activity_name=activity_name, thrs=thrs)

    else :
        traces_selected = []

    # Create an empty hash map, large as the maximum number of traces possible (TO EDIT)
    idx_list = X_train[case_id_name].unique()
    max_traces = len(idx_list)
    traces_hash = HashTable(max_traces)
    for trace_idx in idx_list:

        trace = X_train[X_train[case_id_name] == trace_idx].reset_index(drop=True)
        prev_trace = []

        if thrs == 0 or str_list(trace[activity_name]) in traces_selected:
            for idx in range(len(trace)):

                curr_act = trace[activity_name][idx]
                if idx > 0:
                    if traces_hash.get_val(str_list(prev_trace)) == 'No record found':
                        traces_hash.set_val(str_list(prev_trace), [curr_act])

                    else:
                        if curr_act not in traces_hash.get_val(str_list(prev_trace)):
                            traces_hash.get_val(str_list(prev_trace)).append(curr_act)

                prev_trace.append(curr_act)

    return traces_hash
