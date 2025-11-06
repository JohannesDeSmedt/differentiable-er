import math
import networkx as nx

class BackgroundModel:

    def __init__(self):
        self.number_of_events = 0
        self.number_of_traces = 0
        self.trace_frequency = {}
        self.labels = set()
        self.large_string = ''
        self.lprob = 0
        self.trace_size = {}
        self.log2_of_model_probability = {}
        self.total_number_non_fitting_traces = 0
        pass

    def open_trace(self):
        self.lprob = 0
        self.large_string = ''

    def process_event(self, event_label, probability):
        self.large_string += event_label
        self.number_of_events += 1
        self.labels.add(event_label)
        self.lprob += probability

    def close_trace(self, trace_length, fitting, final_state_prob):
        # print('Closing:', self.large_string)
        self.trace_size[self.large_string] = trace_length
        # print('Trace size:', trace_length)
        self.number_of_traces += 1
        if fitting:
            self.log2_of_model_probability[self.large_string] = (self.lprob + final_state_prob) / math.log(2)
        else:
            self.total_number_non_fitting_traces += 1
        tf = 0
        if self.large_string in self.trace_frequency.keys():
            tf = self.trace_frequency[self.large_string]
        self.trace_frequency[self.large_string] = tf + 1

    def h_0(self, accumulated_rho, total_number_of_traces):
        if accumulated_rho == 0 or accumulated_rho == total_number_of_traces:
            return 0
        else:
            p = (accumulated_rho / total_number_of_traces)
            return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    def compute_relevance(self):
        accumulated_rho = 0
        accumulated_cost_bits = 0
        accumulated_temp_cost_bits = 0
        accumulated_prob_fitting_traces = 0

        for trace_string, trace_freq in self.trace_frequency.items():
            cost_bits = 0
            nftrace_cost_bits = 0

            if trace_string in self.log2_of_model_probability:
                cost_bits = - self.log2_of_model_probability[trace_string]
                accumulated_rho += trace_freq
            else:
                cost_bits = (1 + self.trace_size[trace_string]) * math.log2(1 + len(self.labels))
                nftrace_cost_bits += trace_freq

            accumulated_temp_cost_bits += nftrace_cost_bits * trace_freq
            accumulated_cost_bits += (cost_bits * trace_freq) / self.number_of_traces

            if trace_string in self.log2_of_model_probability:
                accumulated_prob_fitting_traces += trace_freq / self.number_of_traces

        entropic_relevance = self.h_0(accumulated_rho, self.number_of_traces) + accumulated_cost_bits
        return entropic_relevance


def convert_dfg_into_automaton(nodes, arcs):
    agg_outgoing_frequency = {}
    node_info = {node['id']: node['label'] for node in nodes}

    sinks = set(node_info.keys())
    sources = list(node_info.keys())

    for arc in arcs:
        if arc['freq'] > 0:
            arc_from = 0
            if arc['from'] in agg_outgoing_frequency.keys():
                arc_from = agg_outgoing_frequency[arc['from']]
            agg_outgoing_frequency[arc['from']] = arc_from + arc['freq']
            sinks.discard(arc['from'])
            # sources.discard(arc['to'])
            if arc['to'] in sources:
                sources.remove(arc['to'])

    # print('Outgoing frequencies:')
    # print(agg_outgoing_frequency)

    transitions = {}
    for arc in arcs:
        if arc['freq'] > 0:
            if arc['to'] not in sinks:
                label = node_info[arc['to']]
                transitions[(arc['from'], label)] = (arc['to'], arc['freq'] / agg_outgoing_frequency[arc['from']])

    for sink in sinks:
        del node_info[sink]

    states = set()
    outgoing_prob = {}
    trans_table = {}
    for (t_from, label), (t_to, a_prob) in transitions.items():
        trans_table[(t_from, label)] = (t_to, math.log(a_prob))
        states.add(t_from)
        states.add(t_to)
        t_f = 0
        if t_from in outgoing_prob.keys():
            t_f = outgoing_prob[t_from]
        outgoing_prob[t_from] = t_f + a_prob

    final_states = {}
    for state in states:
        if not state in outgoing_prob.keys() or 1.0 - outgoing_prob[state] > 0.000006:
            d_p = 0
            if state in outgoing_prob.keys():
               d_p = outgoing_prob[state]
            final_states[state] = math.log(1 - d_p)

    g = nx.DiGraph()
    data = {}
    for (t_from, label), (t_to, prob) in transitions.items():
        g.add_edge(t_from, t_to, label=label+' - ' + str(round(prob,3)))

    # for start, edge in transitions.items():
    #     print(start, edge)

    # for node in g.nodes:
    #     if g.in_degree(node) == 0:
    #         print(f'{node} has no entries')
    #     if g.out_degree(node) == 0:
    #         print(f'{node} has no exits')
    #         for edge in g.in_edges(node):
    #             print(g.get_edge_data(edge[0], edge[1]))

    # dot = nx.drawing.nx_pydot.to_pydot(g)
    # file_name = './dfgs/temp3'
    # with open(file_name + '.dot', 'w') as file:
    #     file.write(str(dot))
    # check_call(['dot', '-Tpng', file_name + '.dot', '-o', file_name + '.png'])
    # os.remove(file_name + '.dot')

    tR = set()
    for source in sources:
        available = False
        for start, end in transitions.items():
            # print(start, end)
            if source in start or source in end:
                available = True
        if not available:
            tR.add(source)
    for tRe in tR:
        sources.remove(tRe)

    return sources, final_states, trans_table

def sdfa_to_automaton(sdfa, id2label, eps=1e-9, threshold=1e-6):
    """
    Convert a single SDFA matrix (N x N) into:
      sources, final_states, trans_table
    compatible with your DFG evaluator.
    """

    N = sdfa.size(0)

    # Row normalize (avoid division by zero)
    row_sums = sdfa.sum(dim=1, keepdim=True) + eps
    L = sdfa # / row_sums

    trans_table = {}
    outgoing_prob = {i: 0.0 for i in range(N)}

    # Build transitions
    for i in range(N):
        for j in range(N):
            p = L[i, j].item()
            if p > threshold:
                label = id2label[j]
                trans_table[(i, label)] = (j, math.log(p))
                outgoing_prob[i] += p

    # Start states = states with no incoming edges
    all_targets = {t for (_, t) in [v for v in trans_table.values()]}
    sources = [s for s in range(N) if s not in all_targets]

    # Final states = states where probability mass does not sum to 1
    final_states = {}
    for s in range(N):
        residual = 1 - outgoing_prob[s]
        if residual > threshold:
            final_states[s] = math.log(residual)

    return sources, final_states, trans_table

def batch_to_event_log(x_batch, le):
    log = []
    id2label = {i: label for i, label in enumerate(le.classes_)}

    for seq in x_batch:
        events = []
        for idx in seq.tolist():
            if idx == 0:  # padding token
                continue
            label = id2label.get(idx, f"UNK_{idx}")
            events.append({'concept:name': label})
        log.append(events)

    return log


def calculate_entropic_relevance(sdfa, x, le):

    id2label = {i: lbl for i, lbl in enumerate(le.classes_)}
    sources, final_states, trans_table = sdfa_to_automaton(sdfa, id2label)

    log = batch_to_event_log(x, le)

    # assert len(sources) == 1
    ers = []
    for source in sources:
        info_gatherer = BackgroundModel()

        initial_state = source
        for t, trace in enumerate(log):
            # print('trace:', trace)
            curr = initial_state
            non_fitting = False
            info_gatherer.open_trace()
            len_trace = 0
            # print('Current state:', curr)
            for event in trace:
                label = event['concept:name']

                # print(label)
                if label in ['SOC', 'EOC']:
                    continue
                len_trace += 1
                prob = 0

                if not non_fitting and (curr, label) in trans_table.keys():
                    curr, prob = trans_table[(curr, label)]
                else:
                    # print('Not fitting at ', event['concept:name'])
                    # print('Trace:\n')
                    # string_p = ''
                    # for eve in trace:
                    #     string_p += eve['concept:name'] + ' - '
                    # print(string_p)
                    non_fitting = True
                info_gatherer.process_event(label, prob)

            if not non_fitting and curr in final_states.keys():
                info_gatherer.close_trace(len_trace, True, final_states[curr])
            else:
                info_gatherer.close_trace(len_trace, False, 0)

        # print('Non_fitting:', info_gatherer.total_number_non_fitting_traces)
        # print(info_gatherer.number_of_traces)

        entropic_relevance = info_gatherer.compute_relevance()
        ers.append(entropic_relevance)

    entropic_relevance = min(ers)
    # print('Entropic relevance:', entropic_relevance)
    return entropic_relevance