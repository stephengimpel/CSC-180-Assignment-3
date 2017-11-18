/**************************************************
  Neural Network with Choosable Activation & Backpropagation
  --------------------------------------------------
  Adapted from D. Whitley, Colorado State University
  Modifications by S. Gordon
  --------------------------------------------------
  Version 5.0 - July 2017
    scaling removed, activation functions added
  --------------------------------------------------
  compile with g++ nn.c
****************************************************/

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
using namespace std;

// NN parameters  ------------------------------------------------
#define NumINs       3       // number of inputs, not including bias node
#define NumOUTs      1       // number of outputs, not including bias node
#define Criteria     0.5      // all training outputs must be within this for training to stop
#define TestCriteria 0.75    // all testing outputs must be within this for generalization

#define LearningRate 0.185     // most books suggest 0.3 as a starting point
#define Momentum     0.0   // must be >=0 and <1
#define bias         1.0   // output value of bias node (usually 1, sometimes -1 for sigmoid)
#define weightInit   0.6     // weights are initialized randomly with this max magnitude
#define MaxIterate   5000000 // maximum number of iterations before giving up training
#define ReportIntv   10001    // print report every time this many training cases done

// network topology ----------------------------------------------
#define NumNodes1    4       // col 1 - must equal NumINs+1 (extra node is bias node)
#define NumNodes2    10       // col 2 - hidden layer 1, etc.
#define NumNodes3    7       // output layer is last non-zero layer, and must equal NumOUTs
#define NumNodes4    4       // note - last node in input and hidden layers will be used as bias
#define NumNodes5    1       // note - there is no bias node in the output later
#define NumNodes6    0
#define Activation1    0     // use activation=0 for input (first) layer and for unused laters
#define Activation2    1     // Specify desired activation function for hidden and output layers
#define Activation3    2     // 1=sig, 2=tanh, 3=relu, 4=leakyRelu, 5=linear
#define Activation4    1
#define Activation5    2
#define Activation6    0
#define NumOfCols    5       // number of non-zero layers specified above, including the input layer
#define NumOfRows    10       // largest layer - i.e., max number of rows in the network

// data files -----------------------------------------------
#define TrainFile    "Median.dat"  // file containing training data
#define TestFile     "TestingMedian.dat"  // file containing testing data 
#define TrainCases   50           // number of training cases
#define TestCases    15           // number of test cases

// advanced settings ----------------------------------------
#define LeakyReluAmt 0.1


int NumRowsPer[6];          // number of rows used in each column incl. bias
                            // note - bias is not included on output layer
                            // note - leftmost value must equal NumINs+1
                            // note - rightmost value must equal NumOUTs

int ActivationPer[6];

double TrainArray[TrainCases][NumINs + NumOUTs];
double TestArray[TestCases][NumINs + NumOUTs];
int CritrIt = TrainCases;

ifstream train_stream;      // source of training data
ifstream test_stream;       // source of test data

void CalculateInputsAndOutputs ();
void TestInputsAndOutputs();
void TestForward();
void GenReport(int Iteration);
void TrainForward();
void FinReport(int Iteration);

double squashing(double Sum, int whichActiv);
double Dsquashing(double out, int whichActiv);

double ScaleOutput(double X, int which);
double ScaleDown(double X, int which);
void ScaleCriteria();

struct CellRecord
{
  double Output;
  double Error;
  double Weights[NumOfRows];
  double PrevDelta[NumOfRows];
};

struct CellRecord  CellArray[NumOfRows][NumOfCols];
double Inputs[NumINs];
double DesiredOutputs[NumOUTs];
double extrema[NumINs+NumOUTs][2]; // [0] is low, [1] is high
long   Iteration;
double ScaledCriteria[NumOUTs], ScaledTestCriteria[NumOUTs];

// ************************************************************
//  Get data from Training and Testing Files, put into arrays
// ************************************************************
void GetData()
{
  for (int i=0; i < (NumINs+NumOUTs); i++)
  { extrema[i][0]=99999.0; extrema[i][1]=-99999.0;
  }
  // read in training data
  train_stream.open(TrainFile);
  for (int i=0; i < TrainCases; i++)
  { for (int j=0; j < (NumINs+NumOUTs); j++)
    { train_stream >> TrainArray[i][j];
      if (TrainArray[i][j] < extrema[j][0]) extrema[j][0] = TrainArray[i][j];
      if (TrainArray[i][j] > extrema[j][1]) extrema[j][1] = TrainArray[i][j];
  } }
  train_stream.close();

  // read in test data
  test_stream.open(TestFile);
  for (int i=0; i < TestCases; i++)
  { for (int j=0; j < (NumINs+NumOUTs); j++)
    { test_stream >> TestArray[i][j];
      if (TestArray[i][j] < extrema[j][0]) extrema[j][0] = TestArray[i][j];
      if (TestArray[i][j] > extrema[j][1]) extrema[j][1] = TestArray[i][j];
  } }

  // guard against both extrema being equal
  for (int i=0; i < (NumINs+NumOUTs); i++)
  { if (extrema[i][0] == extrema[i][1]) extrema[i][1]=extrema[i][0]+1;
  }
  test_stream.close();

  // scale training and test data to range 0..1
  for (int i=0; i < TrainCases; i++)
  { for (int j=0; j < NumINs+NumOUTs; j++)
    { TrainArray[i][j] = ScaleDown(TrainArray[i][j],j);
  } }
  for (int i=0; i < TestCases; i++)
  { for (int j=0; j < NumINs+NumOUTs; j++)
      TestArray[i][j] = ScaleDown(TestArray[i][j],j);
} }

// **************************************************************
//  Assign the next training pair
// **************************************************************
void CalculateInputsAndOutputs()
{
  static int S=0;
  for (int i=0; i < NumINs; i++) Inputs[i]=TrainArray[S][i];
  for (int i=0; i < NumOUTs; i++) DesiredOutputs[i]=TrainArray[S][i+NumINs];
  S++;
  if (S==TrainCases) S=0;
}

// **************************************************************
//  Assign the next testing pair
// **************************************************************
void TestInputsAndOutputs()
{
  static int S=0;
  for (int i=0; i < NumINs; i++) Inputs[i]=TestArray[S][i];
  for (int i=0; i < NumOUTs; i++) DesiredOutputs[i]=TestArray[S][i+NumINs];
  S++;
  if (S==TestCases) S=0;
}

// *************************   MAIN   *************************************

int main()
{
  int    existsError, ConvergedIterations=0, sizeOfNext;
  long   seedval;
  double Sum, newDelta;

  Iteration=0;

  NumRowsPer[0] = NumNodes1;  ActivationPer[0] = Activation1;
  NumRowsPer[1] = NumNodes2;  ActivationPer[1] = Activation2;
  NumRowsPer[2] = NumNodes3;  ActivationPer[2] = Activation3;
  NumRowsPer[3] = NumNodes4;  ActivationPer[3] = Activation4;
  NumRowsPer[4] = NumNodes5;  ActivationPer[4] = Activation5;
  NumRowsPer[5] = NumNodes6;  ActivationPer[5] = Activation6;

  // initialize the weights to small random values
  // initialize previous changes to 0 (momentum)
  seedval = 555;
  srand(seedval);
  for (int I=1; I < NumOfCols; I++)
  { for (int J=0; J < NumRowsPer[I]-1; J++)
    { for (int K=0; K < NumRowsPer[I-1]; K++)
      { CellArray[J][I].Weights[K] =
          (weightInit*2.0) * ((double)((int)rand() % 100000 / 100000.0)) - weightInit;
        CellArray[J][I].PrevDelta[K] = 0;
  } } }

  GetData();  // read training and test data into arrays
  ScaleCriteria();

  cout << endl << "Iteration     Inputs          ";
  cout << "Desired Outputs          Actual Outputs" << endl;

  // -------------------------------
  // main training loop
  // -------------------------------
  do
  { // retrieve a training pair
    CalculateInputsAndOutputs();
    for (int J=0; J < NumRowsPer[0]-1; J++)
    { CellArray[J][0].Output = Inputs[J];
    }

    //*************************
    //    FORWARD PASS        *
    //*************************

    // hidden layers
    for (int I=1; I < NumOfCols-1; I++)
    { CellArray[NumRowsPer[I-1]-1][I-1].Output = bias;  // bias node at previous layer
      CellArray[NumRowsPer[I-1]-1][I-1].Error = 0.0;    // bias node at previous layer
      for (int J=0; J < NumRowsPer[I]-1; J++)
      { Sum = 0.0;
        for (int K=0; K < NumRowsPer[I-1]; K++)
        { Sum += CellArray[J][I].Weights[K]
               * CellArray[K][I-1].Output;
        }
        CellArray[J][I].Output = squashing(Sum, ActivationPer[I]);
        CellArray[J][I].Error = 0.0;
    } }

    CellArray[NumRowsPer[NumOfCols-2]-1][NumOfCols-2].Output = bias;  // bias feeding output
    CellArray[NumRowsPer[NumOfCols-2]-1][NumOfCols-2].Error = 0.0; 
  
    // output layer
    for (int J=0; J < NumOUTs; J++)
    { Sum = 0.0;
      for (int K=0; K < NumRowsPer[NumOfCols-2]; K++)
      { Sum += CellArray[J][NumOfCols-1].Weights[K]
             * CellArray[K][NumOfCols-2].Output;
      }
      CellArray[J][NumOfCols-1].Output = squashing(Sum, ActivationPer[NumOfCols-1]);
      CellArray[J][NumOfCols-1].Error = 0.0;
    }

    //*************************
    //    BACKWARD PASS       *
    //*************************

    // calculate error at each output node
    for (int J=0; J < NumOUTs; J++)
    { CellArray[J][NumOfCols-1].Error =
        DesiredOutputs[J] - CellArray[J][NumOfCols-1].Output;
    }

    // check to see how many consecutive oks seen so far
    existsError = 0;
    for (int J=0; J < NumOUTs; J++)
    { if (fabs(CellArray[J][NumOfCols-1].Error) > ScaledCriteria[J])
        existsError = 1;
    }
    if (existsError == 0) ConvergedIterations++;
    else ConvergedIterations = 0;

    if (existsError == 1)
    {
      // apply derivative of squashing function to output errors 
      for (int J=0; J < NumOUTs; J++)
      { CellArray[J][NumOfCols-1].Error
         *= Dsquashing(CellArray[J][NumOfCols-1].Output, ActivationPer[NumOfCols-1]);
      }

      // backpropogate errors to hidden layers
      for (int I=NumOfCols-2; I>=1; I--)
      { if (I==NumOfCols-2) sizeOfNext = NumRowsPer[I+1]; else sizeOfNext = NumRowsPer[I+1]-1;
        for (int J=0; J < NumRowsPer[I]; J++)
        { for (int K=0; K < sizeOfNext; K++)
          { CellArray[J][I].Error
            += (CellArray[K][I+1].Weights[J]
              * CellArray[K][I+1].Error);
        } }
        // apply derivative of squashing function to hidden layer errors
        for (int J=0; J < NumRowsPer[I]; J++)
        { CellArray[J][I].Error
          *= Dsquashing(CellArray[J][I].Output, ActivationPer[I]);
      } }

      // adjust weights  of hidden layers 
      for (int I=1; I < NumOfCols-1; I++)
      { for (int J=0; J < NumRowsPer[I]-1; J++)
        { for (int K=0; K < NumRowsPer[I-1]; K++)
          { newDelta = (Momentum * CellArray[J][I].PrevDelta[K])
             + LearningRate * CellArray[K][I-1].Output * CellArray[J][I].Error;
            CellArray[J][I].Weights[K] += newDelta;
            CellArray[J][I].PrevDelta[K] = newDelta;
      } } }
      // adjust weights  of output layer
      for (int J=0; J < NumOUTs; J++)
      { for (int K=0; K < NumRowsPer[NumOfCols-2]; K++)
        { newDelta = (Momentum * CellArray[J][NumOfCols-1].PrevDelta[K])
           + LearningRate * CellArray[K][NumOfCols-2].Output * CellArray[J][NumOfCols-1].Error;
          CellArray[J][NumOfCols-1].Weights[K] += newDelta;
          CellArray[J][NumOfCols-1].PrevDelta[K] = newDelta;
    } } }

    GenReport(Iteration);
    Iteration++;
  } while (!((ConvergedIterations >= CritrIt) || (Iteration >= MaxIterate)));
  // end of main training loop
  // -------------------------------

  FinReport(ConvergedIterations);
  TrainForward();
  TestForward();
  return(0);
}

// *******************************************
//   Run Test Data forward pass only
// *******************************************
void TestForward()
{
  int GoodCount=0;
  double Sum, TotalError=0;
  cout << "Running Test Cases" << endl;
  for (int H=0; H < TestCases; H++)
  { TestInputsAndOutputs();
    for (int J=0; J < NumRowsPer[0]-1; J++) 
    { CellArray[J][0].Output = Inputs[J];
    }
    // hidden layers
    for (int I=1; I < NumOfCols-1; I++)
    { for (int J=0; J < NumRowsPer[I]-1; J++)
      { Sum = 0.0;
        for (int K=0; K < NumRowsPer[I-1]; K++)
        { Sum += CellArray[J][I].Weights[K] 
               * CellArray[K][I-1].Output;
        }
        CellArray[J][I].Output = squashing(Sum, ActivationPer[I]);
        CellArray[J][I].Error = 0.0;
      }
      CellArray[NumRowsPer[I]-1][I].Output = bias;  // bias node
      CellArray[NumRowsPer[I]-1][I].Error = bias;   // error at bias node weight
    }
    // output layer
    for (int J=0; J < NumOUTs; J++)
    { Sum = 0.0;
      for (int K=0; K < NumRowsPer[NumOfCols-2]; K++)
      { Sum += CellArray[J][NumOfCols-1].Weights[K]
             * CellArray[K][NumOfCols-2].Output;
      }
      CellArray[J][NumOfCols-1].Output = squashing(Sum, ActivationPer[NumOfCols-1]);
      CellArray[J][NumOfCols-1].Error = 
        DesiredOutputs[J] - CellArray[J][NumOfCols-1].Output;
      if (fabs(CellArray[J][NumOfCols-1].Error) <= ScaledTestCriteria[J])
        GoodCount++;
      TotalError += CellArray[J][NumOfCols-1].Error *
                    CellArray[J][NumOfCols-1].Error;
    }
    GenReport(-1);
  }
  cout << endl;
  cout << "Sum Squared Error for Testing cases   = " << TotalError << endl;
  cout << "% of Testing Cases that meet criteria = " <<
              ((((double)GoodCount/(double)TestCases)) / (double)NumOUTs);
  cout << endl;
  cout << endl;
}

// ******************************************************
//   Run Training Data forward pass only, after training
// ******************************************************
void TrainForward()
{
  int GoodCount=0;
  double Sum, TotalError=0;
  cout << endl << "Confirm Training Cases" << endl;
  for (int H=0; H < TrainCases; H++)
  { CalculateInputsAndOutputs();
    for (int J=0; J < NumRowsPer[0]-1; J++)
    { CellArray[J][0].Output = Inputs[J];
    }
    // hidden layers
    for (int I=1; I < NumOfCols-1; I++)
    { for (int J=0; J < NumRowsPer[I]-1; J++)
      { Sum = 0.0;
        for (int K=0; K < NumRowsPer[I-1]; K++)
        { Sum += CellArray[J][I].Weights[K] * CellArray[K][I-1].Output;
        }
        CellArray[J][I].Output = squashing(Sum, ActivationPer[I]);
        CellArray[J][I].Error = 0.0;
    } }
    // output layer
    for (int J=0; J < NumOUTs; J++)
    { Sum = 0.0;
      for (int K=0; K < NumRowsPer[NumOfCols-2]; K++)
      { Sum += CellArray[J][NumOfCols-1].Weights[K]
             * CellArray[K][NumOfCols-2].Output;
      }
      CellArray[J][NumOfCols-1].Output = squashing(Sum, ActivationPer[NumOfCols-1]);
      CellArray[J][NumOfCols-1].Error =
        DesiredOutputs[J] - CellArray[J][NumOfCols-1].Output;
      if (fabs(CellArray[J][NumOfCols-1].Error) <= ScaledCriteria[J])
        GoodCount++;
      TotalError += CellArray[J][NumOfCols-1].Error *
                    CellArray[J][NumOfCols-1].Error;
    }
    GenReport(-1);
  }
  cout << endl;
  cout << "Sum Squared Error for Training cases   = " << TotalError << endl;
  cout << "% of Training Cases that meet criteria = " <<
              (((double)GoodCount/(double)TrainCases)/(double)NumOUTs) << endl;
  cout << endl;
}

// *******************************************
//   Final Report
// *******************************************
void FinReport(int CIterations)
{
  cout.setf(ios::fixed); cout.setf(ios::showpoint); cout.precision(4);
  if (CIterations<CritrIt) cout << "Failed to train to criteria" << endl;
  else cout << "Converged to within criteria" << endl;
  cout << "Total number of iterations = " << Iteration << endl;
}

// *******************************************
//   Generation Report
//   pass in a -1 if training is over and displaying results
// *******************************************
void GenReport(int Iteration)
{
  int J;
  cout.setf(ios::fixed); cout.setf(ios::showpoint); cout.precision(4);
  if ((Iteration == -1) || ((Iteration % ReportIntv) == 0))
  { if (Iteration != -1) cout << "  " << Iteration << "  ";
    for (J=0; J < NumRowsPer[0]-1; J++)
      cout << " " << ScaleOutput(Inputs[J],J);
    cout << "  ";
    for (J=0; J < NumOUTs; J++) 
      cout << " " << ScaleOutput(DesiredOutputs[J], NumINs+J);
    for (J=0; J < NumOUTs; J++)
      cout << " " << ScaleOutput(CellArray[J][NumOfCols-1].Output, NumINs+J);
    for (J=0; J < NumOUTs; J++)
      cout << "   " << fabs(ScaleOutput(DesiredOutputs[J],NumINs+J)-ScaleOutput(CellArray[J][NumOfCols-1].Output,NumINs+J));
    cout << endl;
  }
}

// *******************************************
//          Scale Desired Output
// *******************************************
double ScaleDown(double X, int output)
{ return .9*(X-extrema[output][0])/(extrema[output][1]-extrema[output][0])+.05;
}

// ********************************************
//         Scale actual output to original range
// ********************************************
double ScaleOutput(double X, int output)
{
  double range = extrema[output][1] - extrema[output][0];
  double scaleUp = ((X-.05)/.9) * range;
  return (extrema[output][0] + scaleUp);
}

// *******************************************
//          Scale criteria
// *******************************************
void ScaleCriteria()
{ int J;
  for (J=0; J < NumOUTs; J++)
    ScaledCriteria[J] = .9*Criteria/(extrema[NumINs+J][1]-extrema[NumINs+J][0]);
  for (J=0; J < NumOUTs; J++)
    ScaledTestCriteria[J] = .9*TestCriteria/(extrema[NumINs+J][1]-extrema[NumINs+J][0]);
}

// **********************************************
//    Activation ("squashing") Function
// **********************************************
double squashing(double Sum, int whichAct)
{ double squash;
  if (whichAct == 0)
  { cout << "Error - activation 0 requested" << endl; squash = 0.0;
  }
  else if (whichAct == 1)           // sigmoid
  { squash = 1.0/(1.0+exp(-Sum));
  }
  else if (whichAct == 2)           // tanh
  { squash = tanh(Sum);
  }
  else if (whichAct == 3)           // relu
  { squash = 0.0;
    if (Sum > 0.0) squash = Sum;
  }
  else if (whichAct == 4)           // leaky relu
  { squash = 0.0;
    if (Sum > 0.0) squash = Sum;
    if (Sum < 0.0) squash = LeakyReluAmt * Sum;
  }
  else if (whichAct == 5)           // linear
  { squash = Sum;
  }
  return squash;
}

// **********************************************
//    Derivative of Squashing Function
// **********************************************
double Dsquashing(double out, int whichAct)
{ double dsquash;
  if (whichAct == 0)
  { cout << "Error - derivative of activation 0 requested" << endl; dsquash=0.0;
  }
  else if (whichAct == 1)                 // sigmoid
  { dsquash = out * (1.0-out);
  }
  else if (whichAct == 2)                 // tanh
  { dsquash = 1.0 - tanh(out) * tanh(out);
  }
  else if (whichAct == 3)                 // relu
  { dsquash = 0.0;
    if (out > 0.0) dsquash = 1.0;
  }
  else if (whichAct == 4)                 // leaky relu
  { dsquash = 0.0;
    if (out > 0.0) dsquash = 1.0;
    if (out < 0.0) dsquash = LeakyReluAmt;
  }
  else if (whichAct == 5)                 // linear
  { dsquash = 1.0;
  }
  return dsquash;
}

