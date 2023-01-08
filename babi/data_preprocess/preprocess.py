"""
bAbI task reader from https://github.com/siddk/entity-network
adapted for python3 and other needs.

reader.py

Core script containing preprocessing logic - reads bAbI Task Story, and returns
vectorized forms of the stories, questions, and answers.
"""
import numpy as np
import os
import pickle
import re

from functools import reduce

FORMAT_STR = "qa{}_"
PAD_ID = 0
SPLIT_RE = re.compile('(\W+)')


def parse_all(data_path, task_ids, word2id=None, bsz=32, DATA_TYPES=['train', 'valid', 'test'], use_cache=True):
    vectorized_data, story_data, global_story_max, global_query_max = [], [], 0, 0
    for data_type in DATA_TYPES:
        print("read {} ...".format(data_type))
        cache_path = data_path + "-pik/" + FORMAT_STR.format("all") + data_type + ".pik"
        if os.path.exists(cache_path) and use_cache:
            print("accessing cache_path: ", cache_path)
            with open(cache_path, 'rb') as f:
                vectorized_data.append(pickle.load(f))
        else:
            astories = []
            for task_id in task_ids:
                print(f"===========\nload task {task_id}")
                filenames = list(
                    filter(lambda x: FORMAT_STR.format(task_id) in x and data_type in x, os.listdir(data_path)))
                if len(filenames) == 0:
                    print("filename not found for in listdir for {} and {}".format(task_id, data_type))
                    print("skipping ... ")
                    continue
                stories, story_max, query_max, word2id = parse_stories(os.path.join(data_path, filenames[0]), word2id)
                astories.extend(stories)
                global_query_max = max(global_query_max, query_max)
                global_story_max = max(global_story_max, story_max)
            print(f"{data_type} len: {len(astories)}")
            story_data.append(astories)

    if vectorized_data:
        return vectorized_data + [vectorized_data[0][4]]
    else:
        for i, data_type in enumerate(DATA_TYPES):
            print("vectorize {} ...".format(data_type))
            print(f"len dic: {len(word2id)}")
            cache_dir = data_path + "-pik/"
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            cache_path = cache_dir + FORMAT_STR.format("all") + data_type + ".pik"
            S, S_len, Q, A = vectorize_stories(story_data[i], global_story_max, global_query_max, word2id, -1)
            n = int((S.shape[0] / bsz) * bsz)
            with open(cache_path, 'wb') as f:
                pickle.dump((S[:n], S_len[:n], Q[:n], A[:n], word2id), f)
            vectorized_data.append((S[:n], S_len[:n], Q[:n], A[:n], word2id))
        return vectorized_data + [word2id]

def parse(data_path, task_id, word2id=None, bsz=32, DATA_TYPES=['train', 'valid', 'test'], use_cache=True, cache_dir_ext=""):

    vectorized_data, story_data, global_story_max, global_query_max = [], [], 0, 0
    for data_type in DATA_TYPES:
        print("read {} ...".format(data_type))
        cache_path = data_path + f"-pik{cache_dir_ext}/" + FORMAT_STR.format(task_id) + data_type + ".pik"
        if os.path.exists(cache_path) and use_cache:
            print("accessing cache_path: ", cache_path)
            with open(cache_path, 'rb') as f:
                vectorized_data.append(pickle.load(f))
        else:
            filenames = list(
                filter(lambda x: FORMAT_STR.format(task_id) in x and data_type in x, os.listdir(data_path)))
            if len(filenames) == 0:
                print("filename not found for in listdir for {} and {}".format(task_id, data_type))
                print("skipping ... ")
                continue
            stories, story_max, query_max, word2id = parse_stories(os.path.join(data_path, filenames[0]), word2id)
            story_data.append(stories)
            global_query_max = max(global_query_max, query_max)
            global_story_max = max(global_story_max, story_max)
    if vectorized_data:
        return vectorized_data + [vectorized_data[0][4]]
    else:
        for i, data_type in enumerate(DATA_TYPES):
            print(i, data_type)
            print("vectorize {} ...".format(data_type))
            cache_dir = data_path + f"-pik{cache_dir_ext}/"
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            cache_path = cache_dir + FORMAT_STR.format(task_id) + data_type + ".pik"
            S, S_len, Q, A = vectorize_stories(story_data[i], global_story_max, global_query_max, word2id, task_id)
            n = int((S.shape[0] / bsz) * bsz)
            with open(cache_path, 'wb') as f:
                pickle.dump((S[:n], S_len[:n], Q[:n], A[:n], word2id), f)
            vectorized_data.append((S[:n], S_len[:n], Q[:n], A[:n], word2id))
        return vectorized_data + [word2id]


def parse_stories(filename, word2id=None):
    # Open file, get lines
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Go through lines, building story sets
    print("go through lines")
    stories, story = [], []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            query, answer, supporting = line.split('\t')
            query = tokenize(query)
            substory = [x for x in story if x]
            stories.append((substory, query, answer.lower()))
            story.append('')
        else:
            sentence = tokenize(line)
            story.extend(sentence)
    # Build Vocabulary
    print("build vocab")
    if not word2id:
        vocab = set(reduce(lambda x, y: x + y, [q for (_, q, _) in stories]))
        vocab.update(set(reduce(lambda x, y: x + y, [s for (s, _, _) in stories])))
        vocab.update(set(reduce(lambda x, y: x + y, [q for (_, q, _) in stories])))
        print("reduce done!")
        # for (s, _, _) in stories:
        #     for word in s:
        #         vocab.update(word)
        for (_, _, a) in stories:
            vocab.add(a)
        id2word = ['PAD_ID'] + list(vocab)
        word2id = {w: i for i, w in enumerate(id2word)}
    else:
        vocab = set(reduce(lambda x, y: x + y, [q for (_, q, _) in stories]))
        print("reduce done!")
        for (s, _, _) in stories:
            for word in s:
                vocab.update(word)
        for (_, _, a) in stories:
            vocab.add(a)
        id2word = ['PAD_ID'] + list(vocab)
        for v in id2word:
            if v not in word2id:
                word2id[v] = len(word2id)
    # Get Maximum Lengths
    print("get max lengths")
    story_max, query_max = 0, 0
    for (s, q, _) in stories:
        query_max = len(q) if len(q) > query_max else query_max
        story_max = len(s) if len(s) > story_max else story_max

    return stories, story_max, query_max, word2id


def vectorize_stories(stories, story_max, query_max, word2id, task_id):
    # Check Story Max
    if task_id == 3 or int(task_id)<0:
        story_max = min(story_max, 650)
    else:
        story_max = min(story_max, 350)

    # Allocate Arrays
    S = np.zeros([len(stories), story_max], dtype=np.int32)
    Q = np.zeros([len(stories), query_max], dtype=np.int32)
    S_len, A = np.zeros([len(stories)], dtype=np.int32), np.zeros([len(stories)], dtype=np.int32)
    # Fill Arrays
    for i, (s, q, a) in enumerate(stories):
        # Check S Length => All but Task 3 are limited to 70 sentences
        if task_id == 3 or int(task_id)<0:
            s = s[-650:]
        else:
            s = s[-350:]
        # Populate story
        for j in range(len(s)):
            S[i][j] = word2id[s[j]]

        # Populate story length
        S_len[i] = len(s)

        # Populate Question
        for j in range(len(q)):
            Q[i][j] = word2id[q[j]]

        # Populate Answer
        A[i] = word2id[a]

    return S, S_len, Q, A


def tokenize(sentence):
    """
    Tokenize a string by splitting on non-word characters and stripping whitespace.
    """
    return [token.strip().lower() for token in re.split(SPLIT_RE, sentence) if token.strip()]
