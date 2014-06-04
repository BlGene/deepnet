'''
This program is supposed to provide a way to show information about the neural
net, interface to be based on the the convnet version of this file
'''
#New in 2.7 replaces optparse
import argparse
import numpy as np
import pylab  as pl
import matplotlib.cm as cm

#for use_gpu variable
from choose_matrix_library import * 
from trainer import LoadExperiment,CreateDeepnet,LockGPU,FreeGPU
from util import ReadModel, ReadOperation
from dbm import DBM

from pdb import set_trace

class ShowNet:
    
    def __init__(self,op):
        self.op = op
        
        #Always need the model
        model    = ReadModel(self.op.f)
        
        #These are optional
        train_op = None
        if(self.op.train_op):
            train_op = ReadOperation(self.op.train_op)
        
        eval_op = None
        if(self.op.eval_op):
            eval_op  = ReadOperation(self.op.eval_op )
        
        #Now get the model
        self.model = CreateDeepnet(model,train_op,eval_op)
        
    def plot_cost(self):
        #code taken from make_plots.py        

        def preds(metrics_list):
            y = []
            for metric in metrics_list:
                #count = metric.count
                #y.append( 100*(1- metric.correct_preds/metric.count))
                y.append(metric.error)
            return y
        
        def get_plot(v,skip,label):
            y = v[skip:]
            x = np.arange(skip,len(y))
            return pl.plot(x,y,label = label)
        
        skip = 0
        pl.figure(1)
        p1 = get_plot( preds(self.model.net.train_stats), skip, 'train')
        p2 = get_plot( preds(self.model.net.validation_stats), skip, 'valid')
        p3 = get_plot( preds(self.model.net.test_stats), skip, 'test')
        
        pl.legend()
        pl.xlabel('Iterations')
        pl.ylabel('Error %')
        pl.draw()
        
        set_trace()

    def plot_predictions(self):
        '''plots the wrong predictions at the moment, will be extend to plotting random
        predictions later'''

        assert self.model.t_op is not None, 't_op is None.'
        assert self.model.e_op is not None, 'e_op is None.'
        
        print(len(self.model.net.test_stats))
        
        #When the model is trained the label layer is an input layer, change this
        #to an output layer in order to be able to generate category predictions
        for layer in self.model.net.layer:
            if layer.name == u'label_layer':
                print("Reset Layer")
                layer.is_output = True
        
        self.model.SetUpTrainer()
        

        #Create an Analysis class as a functor, just a way to pull out the
        #predictions and targets values from deepnets Evaluate without changing 
        #too much code
        class Analysis():
            def __call__(self,predictions,targets):
                self.predictions = predictions
                self.targets    = targets
            
        ana_obj = Analysis()
        ret = self.model.Evaluate(validation=False, 
                                  collect_predictions=True,
                                  analysis=ana_obj)
        
        predictions = ana_obj.predictions
        targets     = ana_obj.targets[:,0]
        
       
        #Constants controlling display options
        NUM_ROWS = 2
        NUM_COLS = 4
        NUM_IMGS = NUM_ROWS*NUM_COLS
        NUM_TOP_CLASSES = 4
        label_names = [str(i) for i in range(10)]

        #Now that we have extracted the predictions and targets we have everything
        #to analyse the results
        res = np.argmax(predictions,axis=1) == targets
        
        correct_count = np.sum(res)
        
        #position of wrong test samples
        wrong_pos = list(np.argwhere(res ==False).flatten())
        
        #the (wrong) predictions that were made
        preds = predictions[wrong_pos]
        true_label = targets[wrong_pos]

        
        #Iterate through the data handler to get all the wrong predictions
        offset = 0
        batchsize = self.model.test_data_handler.batchsize
        data  = []
        while len(wrong_pos) > 0:
            #print(offset)
            data_list = self.model.test_data_handler.Get()
            
            while wrong_pos[0] < offset + batchsize:
                data.append( data_list[0].asarray()[:,wrong_pos[0]-offset] )
                wrong_pos = wrong_pos[1:]
                if(len(wrong_pos)==0):
                    break
                
            offset += batchsize
        

        #hackety hack
        self.only_errors = True
        
        fig = pl.figure(3)
        fig.text(.4, .95, '%s test case predictions' % ('Mistaken' if self.only_errors else 'Random'))
        
        for r in xrange(NUM_ROWS):
            for c in xrange(NUM_COLS):
                img_idx = r * NUM_COLS + c
                
                if img_idx >= len(data):
                    break
                
                img = data[img_idx]
                
                #specifically mnist here
                img.shape = 28,28

                pl.subplot(NUM_ROWS*2, NUM_COLS, r * 2 * NUM_COLS + c + 1)
                pl.xticks([])
                pl.yticks([])
                

                pl.imshow(img,interpolation='nearest',cmap = pl.cm.binary)
                loc_true_label = int( true_label[img_idx] )
                img_labels = sorted(zip(preds[img_idx,:], label_names), key=lambda x: x[0])[-NUM_TOP_CLASSES:]
                pl.subplot(NUM_ROWS*2, NUM_COLS, (r * 2 + 1) * NUM_COLS + c + 1, aspect='equal')
                ylocs = np.array(range(NUM_TOP_CLASSES)) + 0.5
                height = 0.5
                width = max(ylocs)
                pl.barh(ylocs, [l[0]*width for l in img_labels], height=height, \
                color=['r' if l[1] == label_names[loc_true_label] else 'b' for l in img_labels])
                pl.title(label_names[loc_true_label])
                pl.yticks(ylocs + height/2, [l[1] for l in img_labels])
                pl.xticks([width/2.0, width], ['50%', ''])
                pl.ylim(0, ylocs[-1] + height*2)



        set_trace()
        print(len(self.model.net.test_stats))


    def start(self):
       if self.op.show_cost:
            self.plot_cost()

       if self.op.show_preds:
           self.plot_predictions()

 
        
    @classmethod
    def get_options_parsers(self):
        op = argparse.ArgumentParser()
        
        #These specify the input files
        op.add_argument("-f","-model",action="store",required=True)
        op.add_argument("-train-op",action="store",default=True)
        op.add_argument("-eval-op",action="store",default=None)
        
        #Sepecfiy the actions we should do
        op.add_argument("--show-cost",action='store_true',
                        help="Show specified objective function")
        
        op.add_argument("--show-preds",action='store_true',
                        help="Show predictions made on test set")
        


        return op

if __name__ == '__main__':
  import sys
  if use_gpu == 'yes':
    board = LockGPU()


  op = ShowNet.get_options_parsers()
  
  model  = ShowNet(op.parse_args())
  model.start()


  #model, train_op, eval_op = LoadExperiment(sys.argv[1], sys.argv[2],
  #                                          sys.argv[3])
  #model = CreateDeepnet(model, train_op, eval_op)
  

  if use_gpu == 'yes':
    FreeGPU(board)
