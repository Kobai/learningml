import { Component } from '@angular/core';
import { NavController } from 'ionic-angular';
import { ToastController } from 'ionic-angular';
import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'page-home',
  templateUrl: 'home.html'
})
export class HomePage {
  var1: number;
  var2: number;
  var3: number;
  var4: number;
  model: tf.Model;
  prediction: any;
  flower: any;
  
  constructor(public navCtrl: NavController, private toastCtrl: ToastController) {

  }
  ngOnInit(){
    this.loadModel();
  }
  async loadModel(){
    this.model = await tf.loadModel('../../../assets/nn/model.json');
  }
  async predict(){
    if (this.var1 && this.var2 && this.var3 && this.var4){
     const output = this.model.predict(tf.tensor2d([this.var1,this.var2,this.var3,this.var4],[1,4])) as any;
     const process = Array.from(output.dataSync()).map(Number);
     let index = process.indexOf(Math.max(...process))
     const names = ['Iris Setosa','Iris Versicolor', 'Iris Virginica'];
     const pictures = ['http://www.twofrog.com/images/iris38a.jpg',
                       'https://www.plant-world-seeds.com/images/item_images/000/003/884/large_square/IRIS_VERSICOLOR.JPG?1495391088',
                       'https://www.fs.fed.us/wildflowers/beauty/iris/Blue_Flag/images/iris_virginica_virginica_lg.jpg'];
     this.prediction = names[index];
     this.flower = pictures[index].toString();
    }   
    else{
      this.errorToast();
    }
  }
 errorToast(){
  let toast = this.toastCtrl.create({
    message: 'Error: Not All Fields are Filled',
    duration: 3000,
    position: 'bottom'
  });
  toast.present();
 } 
}
