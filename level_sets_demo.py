#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vyukovy demonstrator metod Level Sets (Chan-Vese a morfologicky GAC).

Ucel: nazorne zobrazit implicitni krivku phi=0, jeji pohyb a kontrolu segmentace.
Neni urcen pro klinicke nebo produkcni vyhodnocovani snimku.

Priklady spusteni:
  python level_sets_demo.py --image blobs --method both --iterations 100
  python level_sets_demo.py --image weak-edge --method cv --init checkerboard --mu 0.35
  python level_sets_demo.py --image fiji-bridge --method both --iterations 80
  python level_sets_demo.py --image cesta/k/obrazu.png --method cv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import csv
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import io, color, img_as_float
from skimage.segmentation import (
    chan_vese, morphological_geodesic_active_contour,
    inverse_gaussian_gradient, checkerboard_level_set,
)
from skimage.transform import resize

ROOT = Path(__file__).resolve().parent


def make_image(kind: str, size: int = 176, seed: int = 2026):
    """Vrati intenzity [0,1], referencni masku nebo None, slovni poznamku."""
    rng = np.random.default_rng(seed)
    if kind in ('fiji-clean', 'fiji-bridge'):
        name = 'castice_ciste.png' if kind == 'fiji-clean' else 'castice_spojene.png'
        full = np.asarray(io.imread(ROOT / 'data' / name))
        if full.ndim == 3:
            full = color.rgb2gray(full[..., :3])
        full = img_as_float(full)
        # Jen pro rychlou demonstraci: jeden ze tri pixelu v obou smerich.
        # Proto neni dovoleno z techto nahledu odvozovat plochy v mm^2 z puvodniho PL09.
        small = full[::3, ::3]
        return small, (small < 0.5), 'Zmenseny vyukovy nahled dat Fiji PL09 (1/3 v kazde ose).'
    if Path(kind).is_file():
        arr = io.imread(kind)
        if arr.ndim == 3:
            arr = color.rgb2gray(arr[..., :3])
        arr = img_as_float(arr)
        # Zachovani pomeru stran pro omezeni vypocetniho casu.
        scale = min(1., 256 / max(arr.shape))
        if scale < 1:
            arr = resize(arr, tuple(max(12,round(x*scale)) for x in arr.shape),
                         preserve_range=True, anti_aliasing=True)
        arr = np.asarray(arr,dtype=float)
        lo, hi = float(np.min(arr)), float(np.max(arr))
        return ((arr - lo) / (hi-lo) if hi>lo else np.zeros_like(arr)), None, 'Vlastni obraz; referencni maska neni znama.'
    yy,xx = np.mgrid[:size,:size]
    X = xx / (size-1)
    Y = yy / (size-1)
    if kind == 'blobs':
        a = ((X-0.34)/0.16)**2 + ((Y-0.53)/0.20)**2 <= 1
        b = ((X-0.71)/0.13)**2 + ((Y-0.45)/0.17)**2 <= 1
        truth = a|b
        bg = 0.76 + 0.10*X - 0.07*Y
        raw = bg - 0.55*truth.astype(float)
        image = ndi.gaussian_filter(raw, 1.4) + rng.normal(0,.028,(size,size))
        note = 'Dva tmave objekty, lehce zasumeny a pozvolne se menici jas pozadi.'
    elif kind == 'weak-edge':
        th = np.arctan2(Y-.50, X-.50)
        rr = np.sqrt(((X-.50)/.29)**2+((Y-.50)/.25)**2)
        truth = rr <= 1 + .12*np.sin(5*th)
        bg = .65 + .09*X - .08*Y
        raw = bg - .21*truth.astype(float)
        image = ndi.gaussian_filter(raw, 3.4) + rng.normal(0,.022,(size,size))
        note = 'Jediny objekt se slabou, neostrou hranou a nerovnomernym pozadim.'
    else:
        raise ValueError('Neznamy priklad. Pouzijte blobs, weak-edge, fiji-clean, fiji-bridge nebo cestu k souboru.')
    return np.clip(image,0,1), truth, note


def signed_distance(mask):
    """Kladne uvnitr; phi=0 predstavuje hranici segmentace."""
    m = np.asarray(mask,dtype=bool)
    return ndi.distance_transform_edt(m)-ndi.distance_transform_edt(~m)


def initial_mask(shape, kind='disk'):
    H,W=shape
    y,x = np.ogrid[:H,:W]
    if kind == 'disk':
        return (x-W/2.)**2+(y-H/2.)**2 <= (min(H,W)*.43)**2
    if kind == 'small-disk':
        return (x-W*.34)**2+(y-H*.53)**2 <= (min(H,W)*.115)**2
    if kind == 'checkerboard':
        return checkerboard_level_set(shape, square_size=18) > 0
    raise ValueError(kind)


def run_chan_vese(image, init='disk', iterations=100, mu=0.25, dt=0.5,
                  snapshot_every=20, lambda1=1., lambda2=1.):
    """Spojity level-set: navrat phi a energii v posloupnosti s kroky segmentace."""
    phi = (np.sin(np.pi*np.arange(image.shape[0])[:,None]/5.) *
           np.sin(np.pi*np.arange(image.shape[1])[None,:]/5.)) if init == 'checkerboard' else signed_distance(initial_mask(image.shape, init))
    shots = [(0, phi.copy())]
    energies=[]
    done=0
    while done < iterations:
        n=min(snapshot_every,iterations-done)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            _, phi, partial = chan_vese(
                image, mu=mu, lambda1=lambda1, lambda2=lambda2,
                dt=dt, tol=1e-10, max_num_iter=n,
                init_level_set=phi, extended_output=True,
            )
        done += len(partial)
        energies.extend(float(x) for x in partial)
        shots.append((done, phi.copy()))
        if len(partial) < n:
            break
    return dict(method='cv', mask=phi>0, phi=phi, shots=shots, energies=np.array(energies))


def run_gac(image, init='disk', iterations=100, sigma=2.0, alpha=100.,
            balloon=-1, smoothing=1, threshold='auto', snapshot_every=20):
    """Morfologicka aproximace geodesic active contour; binarni level set."""
    g = inverse_gaussian_gradient(image, alpha=alpha, sigma=sigma)
    imask = initial_mask(image.shape,init).astype(np.int8)
    shots=[]
    counter=[0]
    def cb(level_set):
        t=counter[0]
        if t==0 or t%snapshot_every==0 or t==iterations:
            shots.append((t,np.array(level_set,dtype=bool,copy=True)))
        counter[0]+=1
    final=morphological_geodesic_active_contour(
        g, num_iter=iterations, init_level_set=imask,
        smoothing=smoothing, balloon=balloon, threshold=threshold,
        iter_callback=cb,
    )
    if not shots or shots[-1][0] != iterations:
        shots.append((iterations,np.array(final,dtype=bool,copy=True)))
    return dict(method='gac',mask=np.array(final,dtype=bool),shots=shots,gimage=g)


def metrics(mask, truth):
    if truth is None:
        return None
    m=np.asarray(mask,dtype=bool); t=np.asarray(truth,dtype=bool)
    intersection=int(np.count_nonzero(m&t))
    union=int(np.count_nonzero(m|t))
    return {'IoU': intersection/union if union else 1.,
            'Dice': 2*intersection/(np.count_nonzero(m)+np.count_nonzero(t)) if (m|t).any() else 1.}


def _draw_contours(ax, image, shots, *, continuous=False, title='', show_legend=True):
    ax.imshow(image,cmap='gray',vmin=0,vmax=1)
    indices = [0, max(0,len(shots)//2),len(shots)-1]
    indices = list(dict.fromkeys(indices))
    colors = ['#ffe168','#49d1f3','#ff4866']
    for i,c in zip(indices,[colors[0],colors[1] if len(indices)==3 else colors[2],colors[2]]):
        step,level=shots[i]
        z=level if continuous else np.asarray(level,float)
        try:
            ax.contour(z,levels=[0.0 if continuous else 0.5],colors=[c],linewidths=1.65)
            if show_legend:
                ax.plot([],[],color=c,lw=2,label=f'iterace {step}')
        except ValueError:
            pass
    ax.set_title(title,fontsize=11)
    if show_legend:
        ax.legend(loc='lower right',fontsize=8,facecolor='#f9f9f9')
    ax.axis('off')


def _plot_cv(image,r,out):
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    _draw_contours(axes[0,0],image,r['shots'],continuous=True,title='Chan-Vese: vývoj nulové hladiny φ=0')
    a=axes[0,1]
    mx=np.percentile(np.abs(r['phi']),96)
    h=a.imshow(r['phi'],cmap='coolwarm',vmin=-mx,vmax=mx)
    a.contour(r['phi'],[0.],colors='k',linewidths=1.3)
    a.set_title('Konečná spojitá funkce φ (uvnitř φ>0)')
    a.axis('off')
    fig.colorbar(h,ax=a,shrink=.8,label='φ')
    a=axes[1,0];a.imshow(r['mask'],cmap='gray',vmin=0,vmax=1)
    a.set_title('Segmentovaná oblast: φ>0');a.axis('off')
    a=axes[1,1]
    a.plot(np.arange(1,len(r['energies'])+1),r['energies'],lw=2)
    a.set_xlabel('Iterace');a.set_ylabel('Energie modelu')
    a.set_title('Průběh energie (může záviset na parametrech)')
    a.grid(alpha=.25)
    fig.suptitle('Skutečná level-set metoda: Chan–Vese',fontweight='bold')
    fig.tight_layout()
    fig.savefig(out,dpi=135,bbox_inches='tight')
    plt.close(fig)


def _plot_gac(image,r,out):
    fig,axes=plt.subplots(1,3,figsize=(13,4.6))
    _draw_contours(axes[0],image,r['shots'],continuous=False,title='Morfologická GAC: vývoj hranice')
    axes[1].imshow(r['gimage'],cmap='magma')
    axes[1].set_title('Funkce zastavení na hraně g(I)')
    axes[1].axis('off')
    axes[2].imshow(image,cmap='gray',vmin=0,vmax=1)
    axes[2].contour(r['mask'],[.5],colors=['#ff4866'],linewidths=1.7)
    axes[2].set_title('Výsledná segmentace');axes[2].axis('off')
    fig.suptitle('Morfologická aproximace geodetické aktivní kontury',fontweight='bold')
    fig.tight_layout();fig.savefig(out,dpi=135,bbox_inches='tight');plt.close(fig)


def _plot_comparison(image,truth,rs,out):
    fig,axes=plt.subplots(1,len(rs)+1,figsize=(4.3*(len(rs)+1),4.1))
    axes[0].imshow(image,cmap='gray',vmin=0,vmax=1)
    if truth is not None:
        axes[0].contour(truth,[.5],colors='#46cb79',linewidths=2)
        axes[0].set_title('Obraz a referenční hranice')
    else:
        axes[0].set_title('Vstup bez referenční masky')
    axes[0].axis('off')
    for i,r in enumerate(rs,1):
        axes[i].imshow(image,cmap='gray',vmin=0,vmax=1)
        axes[i].contour(r['mask'],[.5],colors=['#ff4866'],linewidths=1.7)
        mm=metrics(r['mask'],truth)
        label='Chan–Vese' if r['method']=='cv' else 'Morfologická GAC'
        if mm:label+=f"\nDice={mm['Dice']:.3f}; IoU={mm['IoU']:.3f}"
        axes[i].set_title(label);axes[i].axis('off')
    fig.tight_layout();fig.savefig(out,dpi=135,bbox_inches='tight');plt.close(fig)


def run_normal_flow(shape=(176,176),radius=57,steps=60,speed=0.75,dt=0.5):
    """PDE: φ_t + F|grad φ|=0, F>0 vede ke smrštění kladného nitra."""
    H,W=shape
    yy,xx=np.mgrid[:H,:W]
    mask=(xx-W/2)**2+(yy-H/2)**2<radius**2
    phi=signed_distance(mask)
    shots=[(0,phi.copy())]
    for step in range(1,steps+1):
        py,px=np.gradient(phi)
        phi = phi - dt*speed*np.hypot(px,py)
        if step%10==0:
            phi=signed_distance(phi>0)
        if step%15==0 or step==steps:
            shots.append((step,phi.copy()))
    return shots


def _plot_pde(shots,out):
    fig,axes=plt.subplots(1,3,figsize=(12,4.1))
    phi0=shots[0][1]
    a=axes[0]; m=a.imshow(phi0,cmap='coolwarm',vmin=-50,vmax=50)
    a.contour(phi0,[0],colors='k');a.set_title('Implicitní funkce φ(x,y)');a.axis('off')
    fig.colorbar(m,ax=a,shrink=.8)
    a=axes[1];a.imshow(np.zeros_like(phi0),cmap='gray',vmin=0,vmax=1)
    for i,c in zip([0,1,3,4],['#2e86c1','#00b5ad','#ffad43','#fc4055']):
        step,p=shots[i]
        a.contour(p,[0],colors=[c],linewidths=1.9)
        a.plot([],[],color=c,label=f'krok {step}')
    a.set_title('Pohyb nulové hladiny φ=0');a.legend(fontsize=8)
    a.axis('off')
    areas=[int(np.count_nonzero(p>0)) for _,p in shots]
    axes[2].plot([t for t,_ in shots],areas,'o-')
    axes[2].set_xlabel('Iterace');axes[2].set_ylabel('Plocha nitra (pixelů)')
    axes[2].set_title('Rovnoměrné smršťování bez obrazu')
    axes[2].grid(alpha=.2)
    fig.suptitle('Základní Level Set PDE:  ∂φ/∂t + F |∇φ| = 0',fontweight='bold')
    fig.tight_layout();fig.savefig(out,dpi=135,bbox_inches='tight');plt.close(fig)


def run_experiment(image,truth,out,iterations=80):
    """Zmena tuhosti a inicializace, aby bylo videt jejich roli."""
    cases=[('μ = 0.10; checkerboard','checkerboard',.10),('μ = 0.45; checkerboard','checkerboard',.45),
           ('μ = 0.25; malý disk','small-disk',.25),
           ('μ = 0.25; checkerboard','checkerboard',.25)]
    fig,axes=plt.subplots(2,2,figsize=(9,8))
    rows=[]
    for ax,(title,init,mu) in zip(axes.flat,cases):
        r=run_chan_vese(image,init=init,mu=mu,iterations=iterations,snapshot_every=iterations)
        ax.imshow(image,cmap='gray',vmin=0,vmax=1)
        ax.contour(r['mask'],[.5],colors='#ff4866',linewidths=1.5)
        m=metrics(r['mask'],truth)
        ax.set_title(title + (f"\nDice={m['Dice']:.3f}" if m else ''),fontsize=11)
        ax.axis('off')
        rows.append((title,int(r['mask'].sum()),m['Dice'] if m else float('nan')))
    fig.suptitle('Citlivost Chan–Vese: regularizace μ a počáteční kontura',fontweight='bold')
    fig.tight_layout();fig.savefig(out,dpi=135,bbox_inches='tight');plt.close(fig)
    return rows


def run_demo(image_name='blobs',method='both',iterations=100,init='auto',
             mu=.25,dt=.5,balloon=-1,sigma=2.,smoothing=1,
             output_dir=None,experiment=False):
    out=Path(output_dir or ROOT/'vystupy');out.mkdir(parents=True,exist_ok=True)
    image,truth,note=make_image(image_name)
    tag=Path(image_name).stem if Path(image_name).is_file() else image_name
    tag=tag.replace(' ','_')
    results=[]
    if method in ('cv','both'):
        r=run_chan_vese(image,init=('checkerboard' if init=='auto' else init),iterations=iterations,mu=mu,dt=dt)
        _plot_cv(image,r,out/f'{tag}_01_chan_vese.png')
        results.append(r)
    if method in ('gac','both'):
        r=run_gac(image,init=('disk' if init=='auto' else init),iterations=iterations,sigma=sigma,
                  balloon=balloon,smoothing=smoothing)
        _plot_gac(image,r,out/f'{tag}_02_geodesic_gac.png')
        results.append(r)
    _plot_comparison(image,truth,results,out/f'{tag}_03_porovnani.png')
    rows=[]
    for r in results:
        io.imsave(out/f"{tag}_{r['method']}_maska.png", (r['mask'].astype(np.uint8)*255), check_contrast=False)
        if r['method']=='cv':
            np.save(out/f'{tag}_cv_phi.npy',r['phi'])
        m=metrics(r['mask'],truth) or {}
        ncomponents=ndi.label(r['mask'],structure=np.ones((3,3),dtype=int))[1]
        rows.append({'scenar':tag,'metoda':r['method'],'pocet_pixelu':int(r['mask'].sum()),
                     'komponenty_8':int(ncomponents),
                     'dice':m.get('Dice',''),'iou':m.get('IoU','')})
    with (out/f'{tag}_metriky.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys(),delimiter=';')
        writer.writeheader();writer.writerows(rows)
    if experiment:
        ex=run_experiment(image,truth,out/f'{tag}_04_parametry.png')
        print('Experiment (název, plocha, Dice):',ex)
    print('Vstup:',tag,'|',image.shape,'|',note)
    for row in rows:
        print(f"  {row['metoda']}: plochy pix={row['pocet_pixelu']}, ",
              f"komponenty={row['komponenty_8']}, Dice={row['dice']}")
    print('Výstupy:',out)
    return results,rows


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--image',default='blobs',help='blobs | weak-edge | fiji-clean | fiji-bridge | cesta k PNG/JPG')
    p.add_argument('--method',choices=['cv','gac','both'],default='both')
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--init',choices=['auto','disk','small-disk','checkerboard'],default='auto',help='auto = checkerboard pro Chan-Vese, disk pro GAC')
    p.add_argument('--mu',type=float,default=.25,help='Vaha delky hranice Chan-Vese')
    p.add_argument('--dt',type=float,default=.5,help='Casovy krok Chan-Vese')
    p.add_argument('--balloon',type=int,choices=[-1,0,1],default=-1)
    p.add_argument('--sigma',type=float,default=2.0,help='Rozostreni pro funkci zastaveni GAC')
    p.add_argument('--smoothing',type=int,default=1,help='Morfologicke vyhlazeni GAC')
    p.add_argument('--output-dir',default=str(ROOT/'vystupy'))
    p.add_argument('--experiment',action='store_true',help='4 varianty parametru Chan-Vese')
    args=p.parse_args()
    if args.iterations<1 or args.dt<=0 or args.mu<0 or args.sigma<=0 or args.smoothing<0:
        p.error('Kontrolujte iterace >=1, dt>0, mu>=0, sigma>0, smoothing>=0.')
    shots=run_normal_flow()
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    _plot_pde(shots,Path(args.output_dir)/'00_princip_level_set_phi.png')
    run_demo(args.image,args.method,args.iterations,args.init,args.mu,args.dt,
             args.balloon,args.sigma,args.smoothing,args.output_dir,args.experiment)

if __name__=='__main__':
    main()
