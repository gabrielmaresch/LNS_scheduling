% Rotating  Workforce Scheduling Problem

int: d=7; %number of days to be scheduled
set of int: days = 1..d;

int: n=12; %number of workers
set of int: workers = 1..n;

int: s = 3; %number of shifts
set of int: shifts = 1..s; %1: day, 2: afternoon, 3: night, 0: off


array[0..s+1, 1..2] of int: max_min_lengths = array2d(0..s+1, 1..2,[1,3,2,5,2,5,2,4,2,5]);
% min max for free, day, afternoon, nightm total 

int: min_rest = 1;
int: max_rest = 3;

int: min_work = 2;
int: max_work = 6;


function int: next_day(int: day) = if day = d then 1 else day + 1 endif;
function int: prev_day(int: day) = if day = 1 then d else day - 1 endif;



%%%% How many workers are needed per shift and day (part of the problem statement)

%required workers for shift s' on day d'

array[1..d, 1..s] of int: required_workers = array2d(1..d, 1..s, [3 | i in 1..d, j in 1..s]);



constraint assert(
  forall(day in days)(
   sum(shift in shifts)(required_workers[day, shift])<= n),
   "more workers required than employed"
   );
  
% Decision variable
 array[1..n, 1..d, 0..s] of var 0..1: works; % worker n' works shift s' on day d'
% Auxilary variable: On day d', how many consecutive days has worker n' already worked
array[1..n, 1..d] of var 0..d: work_length_until;
array[1..n, 1..d] of var 0..d: rest_length_until;

 
%%% lenght of consecutive workdays 
constraint
  forall(worker in workers)(
    work_length_until[worker,1] =
      max(shift in shifts)(works[worker,1,shift])
  );

constraint
  forall(worker in workers, day in 2..d)(
    work_length_until[worker,day] =
      if max(shift in shifts)(works[worker,day,shift]) = 1 then
        work_length_until[worker,day-1] + 1
      else
        0
      endif
  );

array[1..n, 1..d] of var 0..d: work_length_cyclic;

constraint forall(worker in workers)(
  work_length_cyclic[worker, 1] =
    if (works[worker, 1, 0] = 0 /\ works[worker, d, 0] = 0) then
      work_length_until[worker,d] + 1
    else
      0
    endif
);

constraint
  forall(worker in workers, day in 2..d)(
    work_length_cyclic[worker,day] =
      if works[worker,day, 0] = 0 then
        work_length_cyclic[worker,day-1] + 1
      else
        0
      endif
  );
  

%%% lenght of consecutive restdays
constraint
  forall(worker in workers)(
    rest_length_until[worker,1] = works[worker, 1, 0]
  );

constraint
  forall(worker in workers, day in 2..d)(
    rest_length_until[worker,day] =
      if works[worker,day,0] = 1 then
        rest_length_until[worker,day-1] + 1
      else
        0
      endif
  );
  
% min and max worklength constraints
constraint forall(worker in workers)(
  max(day in days)(work_length_cyclic[worker, day]) <= max_work
);
%cyclicity
%constraint forall(worker in workers)(
%);


constraint forall(worker in workers)(
  if exists(day in 1..d-1)(works[worker, day, 0] = 0  /\  works[worker, day + 1, 0] = 1) then
    min(day in 1..d where works[worker, day, 0] = 0  /\ works[worker, day + 1, 0] = 1)( % day is last working day
   work_length_cyclic[worker, day]) >= min_work
  else
    true
  endif
);
constraint forall(worker in workers)(  %last day check
  if works[worker, d, 0] = 0 then
    work_length_cyclic[worker, d] >= min_work
  endif
);

% min and max restlength constraints
constraint forall(worker in workers)(
  max(day in days)(rest_length_until[worker, day]) <= max_rest
);
constraint forall(worker in workers)(
  if exists(day in 1..d-1)(works[worker, day, 0] = 1  /\  works[worker, day + 1, 0] = 0) then
    min(day in 1..d where works[worker, day, 0] = 1  /\ works[worker, day + 1, 0] = 0)( % day is last rest
   rest_length_until[worker, day]) >= min_rest
  else
    true
  endif
);
constraint forall(worker in workers)(  %last day check
  if works[worker, d, 0] = 1 then
    rest_length_until[worker, d] >= min_rest
  endif
);




% required number of shifts must be met
constraint forall(day in days, shift in shifts)(
  sum(worker in workers)(works[worker, day, shift]) = required_workers[day, shift]
  );


% every worker can work only one shift per day
constraint forall(worker in workers, day in days)(
  sum(shift in 0..s)(works[worker, day, shift]) = 1
  );

%constraint forall(worker in workers)(
%);



%dont forget to take care of edge cases when schedule starts anew, sofar schedule isnt cyclic

% Forbidden 2 day sequences 
% ND and NA, AD
constraint forall(worker in workers, day in 1..d)(
  works[worker, day, 3] = 1 ->  (works[worker, next_day(day), 1] = 0 /\ works[worker, next_day(day), 2] = 0)
  );
constraint forall(worker in workers, day in 1..d)(
  works[worker, day, 2] = 1 ->  works[worker, next_day(day), 1] = 0
  );
 
% Forbidden 3 day sequences 
% N-D and N-A, A-D
constraint forall(worker in workers, day in 1..d)(
  works[worker, day, 3] = 1  /\ works[worker, next_day(day), 0] = 1->  (works[worker, next_day(next_day(day)), 1] = 0 /\ works[worker, day+2, 2] = 0 )
  ); 
constraint forall(worker in workers, day in 1..d)(
  works[worker, day, 2] = 1  /\ works[worker, next_day(day), 0] = 1->  works[worker, next_day(next_day(day)), 1] = 0
  ); 
 
 
 
 solve satisfy;
 
output [ show(d), " days\n\n"]; output [ "Worker " ++ show(w) ++ ":\t\t" ++ concat([ if fix(works[w,day,0]) = 1 then "-" else if fix(works[w,day,1]) = 1 then "D" else if fix(works[w,day,2]) = 1 then "A" else "N" endif endif endif | day in days ]) ++ "\n" | w in workers ];