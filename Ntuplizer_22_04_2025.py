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
                    ('lp', 11), ('lm', -11),('nup', 12),('num', -12),('bp', 5), ('bm', -5),
                    #carga positiva
                    ('q1', 2),    # Quark up-type (u) → PDG 2, carga +2/3
                    ('q2', -1),   # Antiquark down-type (anti-d) → PDG -1, carga +1/3
                    ('q3', 4),    # Alternativamente, quark charm (c) → PDG 4, carga +2/3
                    ('q4', -3),   # o antiquark strange (anti-s) → PDG -3, carga +1/3
                    #carga negativa
                    ('q5', -2),    # Antiquark up-type (anti-u) → PDG 2, carga -2/3
                    ('q6', 1),   # Quark down-type (d) → PDG -1, carga -1/3
                    ('q7', -4),    # Alternativamente, antiquark charm (anti-c) → PDG 4, carga -2/3
                    ('q8', 3),   # o quark strange (s) → PDG -3, carga -1/3
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

        #ELIMINAMOS VALORES = 0

        
        ls = [particles[q] for q in ['lp','lm'] if particles[q].E() > 1e-10]
        nus = [particles[q] for q in ['nup', 'num'] if particles[q].E() > 1e-10]
        qs = [particles[q] for q in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8']  if particles[q].E() > 1e-10]

        if len(ls) < 1:
            continue
        
    
        

       #NO HACE FALTA ORGANIZAR LOS QUARKS PORQUE EN LA SEGUNDA POSICIÓN SIEMPRE ESTARA EL ANTIQUARK

        
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
        
        lp = deepcopy(ls[0])
        lp.Boost(-boost_ttbar)

        lp_hat = lp.Vect().Unit()

        
        #sea q1 el anti-quark type
        q1 = qs[1]
        anti_quark = deepcopy(q1)
        anti_quark.Boost(-boost_ttbar)
        anti_hat = anti_quark.Vect().Unit()

        cnr_crn = n_hat.Dot(lp_hat) * r_hat.Dot(anti_hat) - r_hat.Dot(lp_hat) * n_hat.Dot(anti_hat)
        cnk_ckn = n_hat.Dot(lp_hat) * k_hat.Dot(anti_hat) - k_hat.Dot(lp_hat) * n_hat.Dot(anti_hat)
        crk_ckr = r_hat.Dot(lp_hat) * k_hat.Dot(anti_hat) - k_hat.Dot(lp_hat) * r_hat.Dot(anti_hat)

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
