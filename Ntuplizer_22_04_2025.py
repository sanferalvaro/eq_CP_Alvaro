#Version Ntuplizer 22/04/2025
#Entrada de la red neuronal de solo dos quarks uno tipo up y otro down
#Consideramos que las metricas de ckn-cnk se usa solo el Quark-down
import ROOT as r 
import pandas as pd 
from tqdm import tqdm 
from lhereader import LHEReader
import random
import math 
from copy import deepcopy
import numpy as np 
from glob import glob 
from multiprocessing import Pool 

def process_file(fil):
    ret = []
    reader = LHEReader(fil, weight_mode='dict')
    for event in reader:
        # === PESOS ===
        w = event.weights
        sm    = w['rw0000']
        op_dn = w['rw0001']
        op_up = w['rw0002']
        lin = (op_up - op_dn) / 2
        quad = (op_up + op_dn - 2 * sm) / 2
        toret = [sm, lin, quad]

        # === SELECCIÓN DE PARTÍCULAS ===
     
        part_list = [
            ('lp', 11), ('lm', -11), ('nup', 12), ('num', -12), ('bp', 5), ('bm', -5),

            # quarks up-type (primero)
            ('q1', 2),    # u
            ('q2', 4),    # c
            ('q3', -2),   # anti-u
            ('q4', -4),   # anti-c

            # quarks down-type (después)
            ('q5', 1),    # d
            ('q6', 3),    # s
            ('q7', -1),   # anti-d
            ('q8', -3),   # anti-s

            ('tp', 6), ('tm', -6)
        ]

            
        particles = {}
        for label, pdgid in part_list:
            particles[label] = r.TLorentzVector()
            for p in event.particles:
                if p.pdgid == pdgid: 
                    particles[label].SetPxPyPzE(p.px, p.py, p.pz, p.energy)
                    break
        
        # === EMPAREJAMIENTO DE LEPTONES ===
        ls = [particles['lp'], particles['lm']]

        # === EMPAREJAMIENTO DE NEUTRINOS ===
        nus = [particles['nup'], particles['num']]

        # === EMPAREJAMIENTO DE B-JETS ===
        bs = [particles['bp'], particles['bm']]
        #random.shuffle(bs)

        # === EMPAREJAMIENTO DE QUARKS ===
        qs = [particles[q] for q in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8']]
        #print(f"VALORES  q1 = {qs[0].Px()}, {qs[0].Py()}, {qs[0].Pz()}, {qs[0].E()}")
        #print(f"VALORES  q2 = {qs[1].Px()}, {qs[1].Py()}, {qs[1].Pz()}, {qs[1].E()}")
        #print(f"VALORES  q3 = {qs[2].Px()}, {qs[2].Py()}, {qs[2].Pz()}, {qs[2].E()}")
        #print(f"VALORES  q4 = {qs[3].Px()}, {qs[3].Py()}, {qs[3].Pz()}, {qs[3].E()}")
        #print(f"VALORES  q5 = {qs[4].Px()}, {qs[4].Py()}, {qs[4].Pz()}, {qs[4].E()}")
        #print(f"VALORES  q6 = {qs[5].Px()}, {qs[5].Py()}, {qs[5].Pz()}, {qs[5].E()}")
        #print(f"VALORES  q7 = {qs[6].Px()}, {qs[6].Py()}, {qs[6].Pz()}, {qs[6].E()}")   
        #print(f"VALORES  q8 = {qs[7].Px()}, {qs[7].Py()}, {qs[7].Pz()}, {qs[7].E()}")
        #print(f"VALORES  lp = {ls[0].Px()}, {ls[0].Py()}, {ls[0].Pz()}, {ls[0].E()}")
        #print(f"VALORES  lm = {ls[1].Px()}, {ls[1].Py()}, {ls[1].Pz()}, {ls[1].E()}")
        #print(qs)

        
        ls = [particles[q] for q in ['lp','lm'] if particles[q].E() > 1e-20]
        if len(ls) < 1:
            continue
        nus = [particles[q] for q in ['nup', 'num'] if particles[q].E() > 1e-10]
        qs = [particles[q] for q in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8']  if particles[q].E() > 1e-10]

        #print(f"Valores  q1 = {qs[0].Px()}, {qs[0].Py()}, {qs[0].Pz()}, {qs[0].E()}")
        #print(f"Valores  q2 = {qs[1].Px()}, {qs[1].Py()}, {qs[1].Pz()}, {qs[1].E()}")
        #print(f"Valores  lp = {ls[0].Px()}, {ls[0].Py()}, {ls[0].Pz()}, {ls[0].E()}")
        #print(qs)
        
        
    
        

       #NO HACE FALTA ORGANIZAR LOS QUARKS PORQUE EN LA SEGUNDA POSICIÓN SIEMPRE ESTARA EL Down-type
        
        """if random.random() < 0.5:
            random.shuffle(qs)
        else:
            # 50% de los casos: No ,mezclamos
            qs = qs"""
            


        # === NUEVA LISTA DE ENTRADA ===
       
        input_particles = [ls[0], bs[0], bs[1], qs[0], qs[1]]
        for p in input_particles:
            for what in "Px,Py,Pz".split(","):
                toret.append(getattr(p, what)())

        # === MET ===
        met = nus[0]
        toret.extend([met.Px(), met.Py()])

        # === TRANSFORMACIÓN AL SISTEMA DEL TTBAR ===
        ttbar = particles['tp'] + particles['tm']
        boost_ttbar = ttbar.BoostVector()
        tp_rest_frame = deepcopy(particles['tp'])
        tp_rest_frame.Boost(-boost_ttbar)

        # === NUEVO SISTEMA DE REFERENCIA ===
        k_hat = tp_rest_frame.Vect().Unit()
        p_hat = r.TVector3(0, 0, 1)
        
        y = k_hat.Dot(p_hat)
        sign_ = float(np.sign(y))


        #PROBLEMA CON LA DIVISIÓN POR 0

        if abs(y) == 1:
            #print(" Warning: y es ±1, evitando división por cero")
            continue # Se evita la división por cero
    
        rval = math.sqrt(1 - y**2)
        r_hat = sign_ / rval * (p_hat - (k_hat * y))
        n_hat = sign_ / rval * (p_hat.Cross(k_hat))

        # Convertir r_hat y n_hat a TVector3
        r_hat = r.TVector3(r_hat.X(), r_hat.Y(), r_hat.Z())
        n_hat = r.TVector3(n_hat.X(), n_hat.Y(), n_hat.Z())

        # === CAMBIO: SOLO UN LEPTÓN EN EL SISTEMA DEL TTBAR ===
        
        lp=deepcopy(ls[0])
        lp.Boost(-boost_ttbar)
        lp_hat = lp.Vect().Unit()

    
        
        #sea q1 el down-quark type porque lo hemos ordenado
        q1 = qs[1]
        down_quark = deepcopy(q1)
        down_quark.Boost(-boost_ttbar)
        down_hat = down_quark.Vect().Unit()

        cnr_crn = n_hat.Dot(lp_hat) * r_hat.Dot(down_hat) - r_hat.Dot(lp_hat) * n_hat.Dot(down_hat)
        cnk_ckn = n_hat.Dot(lp_hat) * k_hat.Dot(down_hat) - k_hat.Dot(lp_hat) * n_hat.Dot(down_hat)
        crk_ckr = r_hat.Dot(lp_hat) * k_hat.Dot(down_hat) - k_hat.Dot(lp_hat) * r_hat.Dot(down_hat)

        toret.extend([cnr_crn, cnk_ckn, crk_ckr])

        ret.append(toret)

    # === CAMBIO EN LAS COLUMNAS DEL DATAFRAME ===
    cols = ['weight_sm', 'weight_lin', 'weight_quad'] + \
           ['%s_%s' % (part, what) for part in 'l,b1,b2,q1,q2'.split(",") for what in 'px,py,pz'.split(",")] + \
           ['met_px', 'met_py'] + ['control_cnr_crn', 'control_cnk_ckn', 'control_crk_ckr']

    df = pd.DataFrame(ret, columns=cols)
    df.to_hdf(fil.replace("unweighted_events_", "ntuple_").replace('.lhe', '.h5'), 'df')

# === RUTA DE LOS ARCHIVOS DE ENTRADA ===
files = glob("/lhome/ext/uovi123/uovi1231/ttbar_semilep_withDecay_sep27/Events/run_01/unweighted_events.lhe")
pool = Pool(15)
pool.map(process_file, files)
