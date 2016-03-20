def var_dump(prob):

    root = prob.root
    udict = root._probdata.unknowns_dict
    pdict = root._probdata.params_dict
    to_prom_name = root._probdata.to_prom_name 

    comps = root.components(recurse=True)

    for c in comps: 
        print "------------------------------"
        print c.pathname
        print "------------------------------"
        print "    params:"
        for p_name in c.params: 
            print "        %s"%p_name
        print "    unknowns:"
        for u_name in c.unknowns: 
            print "        %s"%u_name
        print 
        print 
