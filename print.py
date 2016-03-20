import sqlitedict
import sys


if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = 'aerostruct'

with open (filename + '.vars', 'r') as myfile:
    lines = myfile.readlines()
    lines[-1] += '\n'

    width = lines[0][:-1]
    variables = [name[:-1] for name in lines[1:]]

print_str = '%5s'
print_tup = ('Itn',)
for name in variables:
    print_str = print_str + '%' + width + 's'
    print_tup = print_tup + (name,)
print print_str % print_tup

db = sqlitedict.SqliteDict(filename + '.db', 'openmdao')

counter = 0
for case_name, case_data in db.iteritems():
    if "metadata" in case_name or "derivs" in case_name:
        continue # don't plot these cases
    
    print_str = '%5i'
    print_tup = (counter,)
    for name in variables:
        print_str = print_str + '%' + width + 'e'
        print_tup = print_tup + (case_data['Unknowns'][name],)
    print print_str % print_tup

    counter += 1
