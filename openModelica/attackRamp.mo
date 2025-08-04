model attackRamp
  // Input
  Modelica.Blocks.Interfaces.RealInput param annotation(
    Placement(transformation(origin = {-120, 12}, extent = {{-20, -20}, {20, 20}}), 
              iconTransformation(origin = {-90, 10}, extent = {{-20, -20}, {20, 20}})));

  // Output
  Modelica.Blocks.Interfaces.RealOutput attackedParam annotation(
    Placement(transformation(origin = {120, 10}, extent = {{-20, -20}, {20, 20}}), 
              iconTransformation(origin = {106, 62}, extent = {{-10, -10}, {10, 10}})));

  // Variabili di stato
  Real attackTime;   // Momento di inizio dell'attacco
  Real duration;     // Durata dell'attacco
  Real rampValue;    // Valore della rampa (accelerazione fittizia in m/s²)
  parameter Real simStep = 1;
  Real myTime;

  // ID per il generatore di numeri casuali
  Integer randomId;

  // Parametro di seme per la generazione del numero casuale
  parameter Integer seed = 123432;

  // Funzione per la generazione di numeri casuali impuri
  impure function random
    input Integer id;
    output Real y;
  end random;
  
initial equation
  myTime = 0.0;

equation
  // Inizializzazione del generatore di numeri casuali
  when initial() then
    randomId = Modelica.Math.Random.Utilities.initializeImpureRandom(seed);
    attackTime = Modelica.Math.Random.Utilities.impureRandom(randomId) * (80 - 5) + 5; // Uniforme tra 5 e 80
    duration = Modelica.Math.Random.Utilities.impureRandom(randomId) * (40 - 5) + 5;   // Uniforme tra 5 e 40
    rampValue = (if Modelica.Math.Random.Utilities.impureRandom(randomId) > 0.5 then 1 else -1) * 
                (Modelica.Math.Random.Utilities.impureRandom(randomId) * (0.5 - 0.1) + 0.1); // Uniforme tra -0.5 e -0.1 oppure tra +0.1 e +0.5
  end when;

  der(myTime) = simStep;

  // Implementazione dell'attacco a rampa
  attackedParam = if (myTime >= attackTime) then //and myTime <= attackTime + duration) then 
                      param + rampValue * (myTime - attackTime) 
                   else param;

annotation(
    uses(Modelica(version = "4.0.0")));
end attackRamp;
