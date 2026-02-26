% Rotating  Workforce Scheduling Problem

int: d = 7; %number of days to be scheduled
set of int: days = 1..d;

int: n = 16; %number of workers
set of int: workers = 1..n;

int: s = 3; %number of shifts
set of int: shifts = 1..s; %1: day, 2: afternoon, 3: night, 0: off


array[0..s+1, 1..2] of int: max_min_lengths = array2d(0..4, 1..2, [0, 7, 0, 7, 0, 7, 0, 7, 0, 7]);
% min max for free, day, afternoon, night total

int: min_rest = 2;
int: max_rest = 4;

int: min_work = 4;
int: max_work = 7;
bool: cyclic = true;

int: num_forbidden_sequences = 0;
% Row width is fixed at 3 (max supported forbidden-sequence length).
% The first index selects the sequence; the second index selects its position in the tuple.
array[1..num_forbidden_sequences] of int: forbidden_sequence_lengths = array1d(1..0, []);
% For length-2 sequences, the 3rd entry is padding (0) and ignored via forbidden_sequence_lengths.
array[1..num_forbidden_sequences, 1..3] of int: forbidden_sequences = array2d(1..0, 1..3, []);


function int: next_day(int: day) = if day = d then 1 else day + 1 endif;
function int: prev_day(int: day) = if day = 1 then d else day - 1 endif;
function int: next_worker(int: worker) = if worker = n then 1 else worker + 1 endif;
function int: prev_worker(int: worker) = if worker = 1 then n else worker - 1 endif;



%%%% How many workers are needed per shift and day (part of the problem statement)

%required workers for shift s' on day d'

array[1..d, 1..s] of int: required_workers = array2d(1..7, 1..3, [5, 5, 4, 4, 4, 3, 4, 4, 3, 4, 4, 3, 4, 4, 4, 4, 4, 4, 3, 4, 4]);



constraint assert(
  forall(day in days)(
   sum(shift in shifts)(required_workers[day, shift])<= n),
   "more workers required than employed"
   );

% Decision variable
array[1..n, 1..d] of var 0..s: shift_of; % shift index for worker/day (0 = off)

% Helpers for formulating schedule rotation:
% (worker, d) is followed by (next_worker(worker), 1).

function var int: next_slot_shift(int: worker, int: day) =
  if day < d then shift_of[worker, day + 1] else shift_of[next_worker(worker), 1] endif;

function var int: next2_slot_shift(int: worker, int: day) =
  if day <= d - 2 then shift_of[worker, day + 2]
  elseif day = d - 1 then shift_of[next_worker(worker), 1]
  else shift_of[next_worker(worker), 2]
  endif;
function var int: prev_slot_shift(int: worker, int: day) =
  if day > 1 then shift_of[worker, day - 1] else shift_of[prev_worker(worker), d] endif;

int: num_fixed_assignments;

% Each fixed assignment row is a triple: [worker, day, shift].
% The 1..3 index is tuple width, not number of assignments.
array[1..num_fixed_assignments, 1..3] of int: fixed_assignments;

% Now we actually fix the assignment (partially)
constraint forall(i in 1..num_fixed_assignments)(
  shift_of[fixed_assignments[i, 1], fixed_assignments[i, 2]] = fixed_assignments[i, 3]
);

% Auxilary variable: On day d', how many consecutive days has worker n' already worked
array[1..n, 1..d] of var 0..d: work_length_until;

% Consecutive run length per exact shift (including 0 = off)
array[1..n, 1..d, 0..s] of var 0..d: shift_length_until;

% Slack variables for soft min/max constraints
array[1..n] of var 0..d: slack_work_min;
array[1..n] of var 0..d: slack_work_max;
array[1..n] of var 0..d: slack_rest_min;
array[1..n] of var 0..d: slack_rest_max;
array[1..n] of var 0..1: slack_offday_min;
array[1..n, 1..s] of var 0..d: slack_shifts_min;
array[1..n, 1..s] of var 0..d: slack_shifts_max;
array[1..d, 1..s] of var 0..n: slack_required_min;
array[1..d, 1..s] of var 0..n: slack_required_max;
var int: slack_forbidden_total;
constraint slack_forbidden_total >= 0;
%
% constraint forall(worker in workers)(
%   sum(day in days)(bool2int(shift_of[worker, day] = 0)) + slack_offday_min[worker] >= 1
% );




%%% lenght of consecutive workdays
constraint
  forall(worker in workers)(
    work_length_until[worker,1] = bool2int(shift_of[worker,1] > 0)
  );

constraint
  forall(worker in workers, day in 2..d)(
    work_length_until[worker,day] =
      if shift_of[worker,day] > 0 then
        work_length_until[worker,day-1] + 1
      else
        0
      endif
  );

array[1..n, 1..d] of var 0..d: work_length_cyclic;

constraint forall(worker in workers)(
  work_length_cyclic[worker, 1] =
    if shift_of[worker, 1] > 0 then
      if prev_slot_shift(worker, 1) > 0 then
        min(d, work_length_until[prev_worker(worker), d] + 1)
      else
        1
      endif
    else
      0
    endif
);

constraint
  forall(worker in workers, day in 2..d)(
    work_length_cyclic[worker,day] =
      if shift_of[worker,day] > 0 then
        min(d, work_length_cyclic[worker,day-1] + 1)
      else
        0
      endif
  );


constraint
  forall(worker in workers, shift in 0..s)(
    shift_length_until[worker,1,shift] = bool2int(shift_of[worker,1] = shift)
  );

constraint
  forall(worker in workers, day in 2..d, shift in 0..s)(
    shift_length_until[worker,day,shift] =
      if shift_of[worker,day] = shift then
        shift_length_until[worker,day-1,shift] + 1
      else
        0
      endif
  );

array[1..n, 1..d, 0..s] of var 0..d: shift_length_cyclic;

constraint
  forall(worker in workers, shift in 0..s)(
    shift_length_cyclic[worker,1,shift] =
      if shift_of[worker,1] = shift then
        if prev_slot_shift(worker,1) = shift then
          min(d, shift_length_until[prev_worker(worker),d,shift] + 1)
        else
          1
        endif
      else
        0
      endif
  );

constraint
  forall(worker in workers, day in 2..d, shift in 0..s)(
    shift_length_cyclic[worker,day,shift] =
      if shift_of[worker,day] = shift then
        min(d, shift_length_cyclic[worker,day-1,shift] + 1)
      else
        0
      endif
  );

% min and max worklength constraints
constraint forall(worker in workers)(
  max(day in days)(work_length_cyclic[worker, day]) <= max_work + slack_work_max[worker]
);

constraint forall(worker in workers)(
  if exists(day in 1..d-1)(shift_of[worker, day] > 0  /\  shift_of[worker, day + 1] = 0) then
    min(day in 1..d-1 where shift_of[worker, day] > 0  /\ shift_of[worker, day + 1] = 0)( % day is last working day
   work_length_cyclic[worker, day]) + slack_work_min[worker] >= min_work
  else
    true
  endif
);
constraint forall(worker in workers)(  %last day check
  if shift_of[worker, d] > 0 /\ (not cyclic \/ next_slot_shift(worker, d) = 0) then
    work_length_cyclic[worker, d] + slack_work_min[worker] >= min_work
  endif
);

% min and max restlength constraints
constraint forall(worker in workers)(
  max(day in days)(shift_length_cyclic[worker, day, 0]) <= max_rest + slack_rest_max[worker]
);
constraint forall(worker in workers)(
  if exists(day in 1..d-1)(shift_of[worker, day] = 0  /\  shift_of[worker, day + 1] > 0) then
    min(day in 1..d-1 where shift_of[worker, day] = 0  /\ shift_of[worker, day + 1] > 0)( % day is last rest
   shift_length_cyclic[worker, day, 0]) + slack_rest_min[worker] >= min_rest
  else
    true
  endif
);
constraint forall(worker in workers)(  %last day check
  if shift_of[worker, d] = 0 /\ (not cyclic \/ next_slot_shift(worker, d) > 0) then
    shift_length_cyclic[worker, d, 0] + slack_rest_min[worker] >= min_rest
  endif
);

% soft per-shift min/max consecutive constraints (working shifts only)
constraint forall(worker in workers, shift in shifts)(
  max(day in days)(shift_length_cyclic[worker, day, shift]) <=
    max_min_lengths[shift, 2] + slack_shifts_max[worker, shift]
);

constraint forall(worker in workers, shift in shifts)(
  if exists(day in 1..d-1)(shift_of[worker, day] = shift /\ shift_of[worker, day + 1] != shift) then
    min(day in 1..d-1 where shift_of[worker, day] = shift /\ shift_of[worker, day + 1] != shift)(
      shift_length_cyclic[worker, day, shift]
    ) + slack_shifts_min[worker, shift] >= max_min_lengths[shift, 1]
  else
    true
  endif
);

constraint forall(worker in workers, shift in shifts)(
  if shift_of[worker, d] = shift /\ (not cyclic \/ next_slot_shift(worker, d) != shift) then
    shift_length_cyclic[worker, d, shift] + slack_shifts_min[worker, shift] >= max_min_lengths[shift, 1]
  endif
);




% required number of shifts as soft constraints
constraint forall(day in days, shift in shifts)(
  sum(worker in workers)(bool2int(shift_of[worker, day] = shift))
  + slack_required_min[day, shift] >= required_workers[day, shift]
);
constraint forall(i in 1..num_forbidden_sequences)(
  forbidden_sequence_lengths[i] = 2 \/ forbidden_sequence_lengths[i] = 3
);

constraint slack_forbidden_total =
  sum(seq_idx in 1..num_forbidden_sequences where forbidden_sequence_lengths[seq_idx] = 2)(
    let {
      int: a = forbidden_sequences[seq_idx, 1],
      int: b = forbidden_sequences[seq_idx, 2]
    } in
    sum(worker in workers, day in days where cyclic \/ day < d)(
      bool2int(
        shift_of[worker, day] = a /\
        (if cyclic then next_slot_shift(worker, day) else shift_of[worker, day + 1] endif) = b
      )
    )
  )
  +
  sum(seq_idx in 1..num_forbidden_sequences where forbidden_sequence_lengths[seq_idx] = 3)(
    let {
      int: a = forbidden_sequences[seq_idx, 1],
      int: b = forbidden_sequences[seq_idx, 2],
      int: c = forbidden_sequences[seq_idx, 3]
    } in
    sum(worker in workers, day in days where cyclic \/ day < d - 1)(
      bool2int(
        shift_of[worker, day] = a /\
        (if cyclic then next_slot_shift(worker, day) else shift_of[worker, day + 1] endif) = b /\
        (if cyclic then next2_slot_shift(worker, day) else shift_of[worker, day + 2] endif) = c
      )
    )
  );

solve
  :: int_search([shift_of[w, day] | w in workers, day in days], first_fail, indomain_min, complete)
  minimize
  sum(worker in workers)(
    slack_work_min[worker] + slack_work_max[worker] +
    slack_rest_min[worker] + slack_rest_max[worker] +
    slack_offday_min[worker]
  )
  + sum(day in days, shift in shifts)(slack_required_min[day, shift]);
%output [ show(d), " days\n\n"];
%output [
%  "Violations (slack totals):\n",
%  "  work_min: ", show(sum(worker in workers)(fix(slack_work_min[worker]))), "\n",
%  "  work_max: ", show(sum(worker in workers)(fix(slack_work_max[worker]))), "\n",
%  "  rest_min: ", show(sum(worker in workers)(fix(slack_rest_min[worker]))), "\n",
%  "  rest_max: ", show(sum(worker in workers)(fix(slack_rest_max[worker]))), "\n",
%  "  shifts_min: ", show(sum(worker in workers, shift in 0..s)(fix(slack_shifts_min[worker, shift]))), "\n",
%  "  shifts_max: ", show(sum(worker in workers, shift in 0..s)(fix(slack_shifts_max[worker, shift]))), "\n",
%  "  required_min: ", show(sum(day in days, shift in shifts)(fix(slack_required_min[day, shift]))), "\n",
%  "  required_max: ", show(sum(day in days, shift in shifts)(fix(slack_required_max[day, shift]))), "\n",
%  "  total: ",
%  show(
%    sum(worker in workers)(
%      fix(slack_work_min[worker]) + fix(slack_work_max[worker]) +
%      fix(slack_rest_min[worker]) + fix(slack_rest_max[worker])
%    ) +
%    sum(worker in workers, shift in 0..s)(
%      fix(slack_shifts_min[worker, shift]) + fix(slack_shifts_max[worker, shift])
%    ) +
%    sum(day in days, shift in shifts)(
%      fix(slack_required_min[day, shift]) + fix(slack_required_max[day, shift])
%    )
%  ),
%  "\n\n"
%];
%output [ "Worker " ++ show(w) ++ ":\t\t" ++ concat([ if fix(shift_of[w,day]) = 0 then "-" else if fix(shift_of[w,day]) = 1 then "D" else if fix(shift_of[w,day]) = 2 then "A" else "N" endif endif endif | day in days ]) ++ "\n" | w in workers ];
