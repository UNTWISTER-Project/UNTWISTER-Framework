model attackPlusConstant
  // Input
  Modelica.Blocks.Interfaces.RealInput param annotation(
    Placement(transformation(origin = {-120, 12}, extent = {{-20, -20}, {20, 20}}), 
              iconTransformation(origin = {-90, 10}, extent = {{-20, -20}, {20, 20}})));

  // Output
  Modelica.Blocks.Interfaces.RealOutput attackedParam annotation(
    Placement(transformation(origin = {120, 10}, extent = {{-20, -20}, {20, 20}}), 
              iconTransformation(origin = {106, 62}, extent = {{-10, -10}, {10, 10}})));

  // Variabili di stato (NON parametri, perché cambiano)
  Real attackTime;   
  Real duration;     
  Real attackValue; 
  parameter Real simStep = 1;
  Real myTime;

  // ID for random number generator
  Integer randomId;

  // Parametro di seme per la generazione del numero casuale
  parameter Integer seed = 68303151;

  // Funzione per la generazione di numeri casuali impuri
  impure function random
    input Integer id;
    output Real y;
  end random;
  
initial equation
  myTime = 0.0; // Inizializza myTime a 0.0 al tempo iniziale
equation
  // Inizializzazione del generatore di numeri casuali
  when initial() then
    randomId = Modelica.Math.Random.Utilities.initializeImpureRandom(seed);
    //Generazione di valori casuali per attackTime, duration e attackValue
    attackTime = Modelica.Math.Random.Utilities.impureRandom(randomId) * (80 - 5) + 5;  // Uniforme tra 5 e 80
    duration = Modelica.Math.Random.Utilities.impureRandom(randomId) * (40 - 5) + 5;     // Uniforme tra 5 e 40
    //attackValue = Modelica.Math.Random.Utilities.impureRandom(randomId) * (5 - 1) + 1;    // Uniforme tra 1 e 5
    attackValue = (if Modelica.Math.Random.Utilities.impureRandom(randomId) > 0.5 then
                 Modelica.Math.Random.Utilities.impureRandom(randomId) * 4 + 1  // tra 1 e 5
               else
                 -(Modelica.Math.Random.Utilities.impureRandom(randomId) * 4 + 1)); // tra -1 e -5  // uniforme tra -1 e -5 o +1 e +5 
    
    end when; 

  der(myTime) = simStep;  // Gestisce il passo temporale per myTime
  //Modelica.Utilities.Streams.print("time: " + String(time) + ", myTime: " + String(myTime));
  
  // Implementazione dell'attacco a scalino
  attackedParam = if (myTime >= attackTime) then //and myTime <= attackTime + duration) then 
    param + attackValue else param;

annotation(
    uses(Modelica(version = "4.0.0")));
end attackPlusConstant;
