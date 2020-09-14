
import csv
import sys
in_file=sys.argv[1]

data = []
with open(in_file) as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        data.append(row)


max_value = max(float(count) for ep,count,lr,name in data)
increment = max_value / 25

longest_label_length = max(len(label) for _,_,_,label in data)

for ep,count,_,label in data:
    count = float(count)
    # The ASCII block elements come in chunks of 8, so we work out how
    # many fractions of 8 we need.
    # https://en.wikipedia.org/wiki/Block_Elements
    bar_chunks, remainder = divmod(int(count * 8 / increment), 8)

    # First draw the full width chunks
    bar = '█' * bar_chunks

    # Then add the fractional part.  The Unicode code points for
    # block elements are (8/8), (7/8), (6/8), ... , so we need to
    # work backwards.
    if remainder > 0:
        bar += chr(ord('█') + (8 - remainder))

    # If the bar is empty, add a left one-eighth block
    bar = bar or  '▏'

    print(f'{ep}\t{label.rjust(longest_label_length)} | {count:#4f} \t {bar}')
