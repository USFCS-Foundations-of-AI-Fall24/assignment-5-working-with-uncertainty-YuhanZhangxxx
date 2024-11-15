import random
import argparse
import os
import math
import re
import sys

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq
        self.outputseq = outputseq

    def __str__(self):
        return ' '.join(self.stateseq) + '\n' + ' '.join(self.outputseq) + '\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)

class HMM:
    def __init__(self, transitions=None, emissions=None):
        self.transitions = transitions if transitions else {}
        self.emissions = emissions if emissions else {}
        self.basename = ''

    def load(self, basename):
        self.transitions = {}
        self.emissions = {}
        self.basename = basename

        with open(basename + '.trans', 'r') as trans_file:
            for line in trans_file:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#'):
                    parts = line[1:].strip().split()
                    if len(parts) != 2:
                        continue
                    to_state, prob = parts
                    from_state = '#'
                else:
                    parts = line.split()
                    if len(parts) != 3:
                        continue
                    from_state, to_state, prob = parts
                prob = float(prob)
                self.transitions.setdefault(from_state, {})[to_state] = prob

        with open(basename + '.emit', 'r') as emit_file:
            for line in emit_file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) != 3:
                    continue
                state, emission, prob = parts
                prob = float(prob)
                self.emissions.setdefault(state, {})[emission] = prob

    def generate(self, n):
        states = []
        emissions = []
        current_state = '#'

        for _ in range(n):
            if current_state not in self.transitions:
                break
            next_states = list(self.transitions[current_state].keys())
            probs = list(self.transitions[current_state].values())
            if sum(probs) == 0:
                break
            current_state = random.choices(next_states, weights=probs, k=1)[0]
            states.append(current_state)

            if current_state not in self.emissions:
                break
            emission_options = list(self.emissions[current_state].keys())
            emission_probs = list(self.emissions[current_state].values())
            if sum(emission_probs) == 0:
                break
            emission = random.choices(emission_options, weights=emission_probs, k=1)[0]
            emissions.append(emission)

        return Sequence(states, emissions)

    def forward(self, sequence):
        states = list(self.emissions.keys())
        alpha = [{}]
        for state in states:
            trans_prob = self.transitions['#'].get(state, 0)
            emit_prob = self.emissions[state].get(sequence[0], 0)
            alpha[0][state] = trans_prob * emit_prob

        for t in range(1, len(sequence)):
            alpha.append({})
            for curr_state in states:
                total = sum(alpha[t - 1][prev_state] * self.transitions[prev_state].get(curr_state, 0)
                             for prev_state in states)
                emit_prob = self.emissions[curr_state].get(sequence[t], 0)
                alpha[t][curr_state] = total * emit_prob

        final_probs = alpha[-1]
        total_prob = sum(final_probs.values())
        print(f"Total probability of the observation sequence: {total_prob}")

        if final_probs:
            most_probable_state = max(final_probs, key=final_probs.get)
            print(f"Most probable final state: {most_probable_state}")
        else:
            print("No valid final states found.")

        if self.basename == 'lander':
            safe_zones = ['2,2', '3,3', '4,4']
            if most_probable_state in safe_zones:
                print("Safe to land.")
            else:
                print("Not safe to land.")

    def viterbi(self, sequence, true_states=None):
        states = list(self.emissions.keys())
        V = [{}]
        path = {}

        for state in states:
            trans_prob = self.transitions['#'].get(state, 0)
            emit_prob = self.emissions[state].get(sequence[0], 0)
            V[0][state] = trans_prob * emit_prob
            path[state] = [state]

        for t in range(1, len(sequence)):
            V.append({})
            new_path = {}

            for curr_state in states:
                max_prob = 0
                prev_st_selected = None
                for prev_state in states:
                    prev_prob = V[t - 1].get(prev_state, 0)
                    if prev_prob == 0:
                        continue

                    trans_p = self.transitions[prev_state].get(curr_state, 0)
                    emit_p = self.emissions[curr_state].get(sequence[t], 0)

                    if trans_p == 0 or emit_p == 0:
                        continue

                    prob = prev_prob * trans_p * emit_p

                    if prob > max_prob:
                        max_prob = prob
                        prev_st_selected = prev_state

                if max_prob > 0:
                    V[t][curr_state] = max_prob
                    new_path[curr_state] = path[prev_st_selected] + [curr_state]

            if not V[t]:
                print(f"No valid paths at time {t}")
                return

            path = new_path

        n = len(sequence) - 1
        max_prob = max(V[n].values())
        possible_states = [state for state in V[n] if V[n][state] == max_prob]
        state = possible_states[0]
        print('Most probable state sequence:')
        print(' '.join(path[state]))

        if true_states:
            if len(true_states) != len(path[state]):
                print("Warning: The length of true states and predicted states does not match.")
                print(f"Length of observation sequence: {len(sequence)}")
                print(f"Length of predicted states: {len(path[state])}")
                print(f"Length of true states sequence: {len(true_states)}")
            else:
                correct = sum(p == t for p, t in zip(path[state], true_states))
                accuracy = correct / len(true_states)
                print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HMM Tool')
    parser.add_argument('basename', help='Base name of the HMM files')
    parser.add_argument('--generate', type=int, help='Generate a sequence of given length')
    parser.add_argument('--generate_only_obs', action='store_true', help='Generate only the observation sequence')
    parser.add_argument('--forward', type=str, help='Perform forward algorithm on given sequence file')
    parser.add_argument('--viterbi', type=str, help='Perform Viterbi algorithm on given sequence file')
    args = parser.parse_args()

    h = HMM()
    h.load(args.basename)

    if args.generate:
        seq = h.generate(args.generate)
        if args.generate_only_obs:
            print(' '.join(seq.outputseq))
        else:
            print('Generated States:')
            print(' '.join(seq.stateseq))
            print('Generated Emissions:')
            print(' '.join(seq.outputseq))

    if args.forward:
        if os.path.exists(args.forward):
            with open(args.forward, 'r') as f:
                sequence = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    sequence.extend(line.split())
            h.forward(sequence)
        else:
            print(f"Observation file {args.forward} not found.")

    if args.viterbi:
        if os.path.exists(args.viterbi):
            with open(args.viterbi, 'r') as f:
                sequence = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tokens = re.findall(r"\w+|[^\s\w]", line)
                    sequence.extend(tokens)

            tagged_file = args.viterbi.replace('.obs', '.tagged.obs')
            true_states = []
            if os.path.exists(tagged_file):
                with open(tagged_file, 'r') as f_tagged:
                    lines = f_tagged.readlines()
                    for i in range(0, len(lines), 2):
                        line = lines[i].strip()
                        if not line:
                            continue
                        true_states.extend(line.split())
            else:
                true_states = None

            h.viterbi(sequence, true_states)
        else:
            print(f"Observation file {args.viterbi} not found.")
