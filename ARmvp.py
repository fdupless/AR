import time,sys,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
if(not sys.version_info[0]<3):
    from importlib import reload
from matplotlib import animation
import scipy.signal as scisig
import pptools as ppt
import matplotlib.colors as colors
import matplotlib.colorbar as cb
reload(ppt)
from skimage import io,data
from skimage import measure as ms

#df=pd.read_csv("PAMAP2_Dataset/Protocol/subject"+str(102)+".dat",delimiter=" ") #dataframe of subject 101
#df=df.fillna(method='ffill',axis=0) #fills missing data with the next valid entry
#walking=activity('walking',df)
#running=activity('running',df)
#standing=activity('standing',df)

subjects=[str(100+i) for i in range(1,8)]+['109']+['101opt','105opt','106opt','109opt']
sub=['108','108opt']
activities=np.array(['lying','sitting','standing','walking','running','cycling','Nordic Walking','watching TV','computer work','car driving','ascending stairs','descending stairs','vacuum cleaning','ironing','folding laundry','house cleaning','playing soccer','rope jumping'])
actlabel=np.array([1,2,3,4,5,6,7,9,10,11,12,13,16,17,18,19,20,24])
#acts=np.array(['lying','sitting','standing','walking'])

tinterval=100 # time interval in s*100 that our classifier will train on
tmp=list([s for s in sys.argv if 'tinterval_' in s])
if(len(tmp)>0):
    tinterval=int(tmp[0][-3:])



class activity:
    def __init__(self,activity,df):
        tmp=np.where(activities==activity)[0]
        if(len(tmp)>0):
            actnum=actlabel[tmp[0]]
            indx=np.where(df['0'].values==actnum)[0] #index of the activity
            self.actname=activity
            self.data=df.values[indx]
            self.ActClass=np.zeros(len(activities))
            self.ActClass[tmp[0]]=1
        #else: comment out, better to get an error and let me know something is wrong
        #    self.data=np.array([]) 

def PPFeats(data):
    time=data[:,0]
    meanHR=np.average(data[:,2])
    
    Hand_Chest_Ankle=[3,20,37]

    feats=[]
    feats.append(meanHR)
    for iSens in Hand_Chest_Ankle:
        mean_T=np.average(data[:,iSens+0])
        Acc=np.array([data[:,iSens+1],data[:,iSens+2],data[:,iSens+3]])
        feats.append(mean_T)
        for j in range(3):
            for element in ppt.ffttospline(Acc[j]):
                feats.append(element)

        Gyro=np.array([data[:,iSens+7],data[:,iSens+8],data[:,iSens+9]])
        for j in range(3):
            for element in ppt.ffttospline(Gyro[j]):
                feats.append(element)

                
        #Magne=np.array([data[:,iSens+10],data[:,iSens+11],data[:,iSens+12]])
        #for j in range(3):
        #    for element in ppt.ffttospline(Magne[j]):
        #        feats.append(element)
    return feats


def gettrainData(subjects,act,q,TimeToAnalyze=tinterval,outputRAW=False):
    trainX=[[] for i in range(len(act))]
    for subID in subjects:
        df=pd.read_csv("PAMAP2_Dataset/Protocol/subject"+subID+".dat",delimiter=" ") #dataframe of subject num
        df=df.fillna(method='ffill',axis=0) #fills missing data with the next valid entry
        df=df.fillna(method='bfill',axis=0) 
        for i in range(len(act)):
            tempdata=activity(act[i],df).data
            totdata=int(np.floor(len(tempdata)/TimeToAnalyze))
            for k in range(totdata):
                trainX[i].append(tempdata[TimeToAnalyze*k:TimeToAnalyze*(k+1)])
    protrainX=[]
    trainY=[]

    if(outputRAW):
        for i in range(len(act)): #sometimes we just want the raw data
            for rawdata in trainX[i]:
                protrainX.append(rawdata)
                trainY.append(actlabel[i])
    else:
        for i in range(len(act)): #but usually we want it preprocessed first
            for rawdata in trainX[i]:
                protrainX.append(PPFeats(rawdata))
                trainY.append(actlabel[i])

    protrainX=np.asarray(protrainX)
    if(q!=1):
        q.put([protrainX,trainY])
    else: return protrainX,trainY

def labtostr(dataY):
    tmp=[]
    for i in range(len(dataY)):
        ind=np.where(actlabel==dataY[i])[0][0]
        tmp.append(activities[ind])
    return np.array(tmp)

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.externals import joblib

def fitlearner(X,Y,acts=activities,classifier='RFC',name='_'):
    
    clf=GSCV(RFC(),{'n_estimators':np.arange(1,5,1)*10,'max_features':["auto","sqrt","log2",None]}) #default classifier
    print('Training the classifier...')
    clf.fit(X,Y)

    testX,testY=gettrainData(['108'],acts,1)
   
    predictions=clf.predict(testX)
    success=(predictions==testY).sum()*1.0/len(predictions)
    print('Success Rate',success)
    _=joblib.dump(clf,'Classifier_'+name)
    return clf

class AR:
    def __init__(self,name='RFC_activity'):
        self.clf=joblib.load(name)     
    
    def displayact(self,data):
        testX=PPFeats(data)
        prediction=self.clf.predict([testX])
        return activities[np.where(actlabel==prediction[0])[0]][0]



#subjects=np.arange(1,8,1)+100
#X,Y=ARmvp.gettrainData(subjects,acts)
#ARmvp.fitlearner(X,Y,acts=acts)


################ Functions for the Simulation movie ##############################

def bogusdata(t):
    df=pd.read_csv("PAMAP2_Dataset/Protocol/subject"+'108'+".dat",delimiter=" ")
    df=df.fillna(method='ffill') #fills missing data with the next valid entry
    df=df.fillna(method='bfill') 
    df2=pd.read_csv("PAMAP2_Dataset/Protocol/subject"+'108opt'+".dat",delimiter=" ")
    df2=df2.fillna(method='ffill') #fills missing data with the next valid entry
    df2=df2.fillna(method='bfill') 

    #testwalk=activity('walking',df)
    #teststand=activity('standing',df)
    #testsit=activity('sitting',df)

    testwalk=activity('vacuum cleaning',df)
    teststand=activity('playing soccer',df2)
    testsit=activity('cycling',df)

    walkt=(t-10)*100
    start=1200
    testdata=np.append(np.append(testsit.data[start:start+500],teststand.data[start:start+500],axis=0),testwalk.data[start:start+walkt],axis=0)
    p=4
    testdataX=np.append(np.append(testsit.data[start:start+500,p],teststand.data[start:start+500,p]),testwalk.data[start:start+walkt,p])
    p=5
    testdataY=np.append(np.append(testsit.data[start:start+500,p],teststand.data[start:start+500,p]),testwalk.data[start:start+walkt,p])
    p=6
    testdataZ=np.append(np.append(testsit.data[start:start+500,p],teststand.data[start:start+500,p]),testwalk.data[start:start+walkt,p])

    timesit=testsit.data[start:start+500,0]-testsit.data[start,0]
    timestand=teststand.data[start:start+500,0]-teststand.data[start,0]+timesit[-1]+0.01
    timewalk=testwalk.data[start:start+walkt,0]-testwalk.data[start,0]+timestand[-1]+0.01
    testdataTime=np.append(np.append(timesit,timestand),timewalk)
    return testdataX,testdataY,testdataZ,testdataTime,testdata


def _blit_draw(self, artists, bg_cache):
    # Handles blitted drawing, which renders only the artists given instead
    # of the entire figure.
    updated_ax = []
    for a in artists:
        # If we haven't cached the background for this axes object, do
        # so now. This might not always be reliable, but it's an attempt
        # to automate the process.
        if a.axes not in bg_cache:
            # bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
            # change here
            bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
        a.axes.draw_artist(a)
        updated_ax.append(a.axes)

    # After rendering all the needed artists, blit each axes individually.
    for ax in set(updated_ax):
        # and here
        # ax.figure.canvas.blit(ax.bbox)
        ax.figure.canvas.blit(ax.figure.bbox)

animation.Animation._blit_draw = _blit_draw

def run_simulation(name='Simulation',clfname='RFC_activities',TimeToAnalyze=tinterval):
    
    global predict
    predict='?'

    clf=AR(name=clfname)
    totaltime=20
    data=bogusdata(totaltime) #hand accelerometer data[0:3], full dataframe data[4]
    
    fig = plt.figure()
    ax = plt.axes(xlim=(0, totaltime), ylim=(-30, 30))
    plt.title('Sample input: Hand Sensor 3D acceleration')
    plt.ylabel(r'a $(m/s^2)$')
    plt.xlabel('time elapsed (s)')
    pixlocx,pixlocy=0.02,0.9
    line1, = ax.plot([], [], lw=2,color='blue')
    line2, = ax.plot([], [], lw=2,color='green')
    line3, = ax.plot([], [], lw=2,color='orange')
    ttl=ax.text(pixlocx,pixlocy,'',transform = ax.transAxes,fontsize=12,bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})
    plt.legend([line3, line2,line1], ['X-axis', 'Y-axis','Z-axis'])
    
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        ttl.set_text('')
        return line1,line2,line3,ttl,
    def animate(i):
        global predict
        x=data[3][:i]
        y1=data[0][:i]
        y2=data[1][:i]
        y3=data[2][:i]
        line1.set_data(x,y1)
        line2.set_data(x,y2)
        line3.set_data(x,y3)
        if((i*1.0/100)==np.floor(i/100) and i>=TimeToAnalyze):
            predict=clf.displayact(data[4][i-TimeToAnalyze:i])
        ttl.set_text('Activity Recognition Ouput: '+predict)
        return line1,line2,line3,ttl,
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=totaltime*100, interval=10, blit=True)
    anim.save(name+'.mp4', fps=100, extra_args=['-vcodec', 'libx264'])
    return

def confusionmatrix(dataX,dataY,name='confmat',clfname='RFC_activity',acts=activities):
    clf=joblib.load(clfname)
    confmat=np.zeros((len(acts),len(acts)))
    for i in range(len(dataX)):
        pred=labtostr(clf.predict([dataX[i],]))[0]
        colint=np.where(activities==pred)[0][0]
        rowint=np.where(activities==dataY[i])[0][0]
        confmat[rowint][colint]+=1
    for irow in range(len(confmat)):
        if(np.max(confmat[irow])>0):
            confmat[irow]/=np.sum(confmat[irow])
 

    colmap=plt.cm.plasma
    cNorm  = colors.Normalize(vmin=0, vmax=1)
    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colmap)
    scalarMap.set_array([0,1])

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(confmat, cmap=colmap, alpha=1)
    
    cbar=fig.colorbar(scalarMap)
    cbar.set_label(r'(# Classification)/ (Total Row # Classification)',rotation=90,size=23)
    cbar.ax.tick_params(labelsize=20)

    # Format
    fig = plt.gcf()
    fig.set_size_inches(len(acts), len(acts)-2)

    # turn off the frame
    ax.set_frame_on(False)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(confmat.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(confmat.shape[1]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # Set the labels
    labels = acts

    ax.set_xticklabels(labels, minor=False,size=20)
    ax.set_yticklabels(labels, minor=False,size=20)

    #     rotate the
    plt.xticks(rotation=90)

    ax.grid(False)

    # Turn off all the ticks
    ax = plt.gca()

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    plt.tight_layout()
    plt.savefig('ConfusionMatrices/ConfusionMatrix')

    return

####### RUNS THE CODE #######
from multiprocessing import Process, Queue
if(__name__=='__main__'): 
    if( 'train' in sys.argv ):
        Xname='X_all'
        Yname='Y_all'
        acts=activities
        t0=time.clock()
        print('Preprocessing the raw data...')
        ncores=1
        tmp=list([s for s in sys.argv if 'ncores_' in s])
        if(len(tmp)>0):
            ncores=int(tmp[0][-1])
        if(ncores>1):
            X=np.array([])
            Y=np.array([])
            q=Queue()
            subprocesses = []
            M=int(np.floor(len(subjects)/ncores))
            for i in range(ncores-1):
                print('Starting core', i+1)
                p=Process(target=gettrainData, args=(subjects[M*i:M*(i+1)],acts,q))
                p.start()
                subprocesses.append(p)
            
            print('Starting core', ncores)
            p=Process(target=gettrainData, args=(subjects[M*(i+1):],acts,q))
            p.start()
            subprocesses.append(p)
            ic=0
            while(len(X)==0):
                tmp=q.get(True)
                ic=1
                X=tmp[0]
                Y=tmp[1]
            for i in range(ic,ncores):
                tmp=q.get(True)
                X=np.append(X,tmp[0],axis=0)
                Y=np.append(Y,tmp[1])
            np.save(Xname,X)
            np.save(Yname,Y)
            while subprocesses:
                subprocesses.pop().join()
        else:
            X,Y=gettrainData(subjects,acts,1)
            np.save(Xname,X)
            np.save(Yname,Y)
        print('Data preprocessed in ', time.clock()-t0,' seconds')

        X=np.load(Xname+'.npy')
        Y=np.load(Yname+'.npy')
        tmp=list([s for s in sys.argv if 'clfname_' in s])
        if(len(tmp)>0):
            clfname=str(tmp[0][8:])
        fitlearner(X,Y,acts=acts,name=clfname)

if( 'do_sim' in sys.argv ):
    run_simulation(clfname='Classifier_1deg2knomag')


