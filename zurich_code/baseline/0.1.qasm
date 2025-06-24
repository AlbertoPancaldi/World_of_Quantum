// Generated from Cirq v1.4.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2)]
qreg q[17];

h q[0];
h q[1];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];



cz q[2], q[5];
cz q[4], q[1];
cz q[3], q[0];
cz q[6], q[7];
cz q[14], q[8];
cz q[15], q[10];
cz q[16], q[13];

h q[0];
h q[5];
h q[7];



cz q[14], q[13];
cz q[5], q[9];
cz q[11], q[8];
cz q[2], q[10];
cz q[3], q[1];


h q[1];
h q[2];
h q[3];
h q[5];
h q[9];
h q[13];




cz q[11], q[3];
cz q[6], q[2];
cz q[4], q[5];
cz q[9], q[8];

h q[3];
h q[6];
h q[8];
h q[9];



cz q[11], q[6];
cz q[4], q[2];
cz q[15], q[9];

h q[2];
h q[4];
h q[6];
h q[15];


cz q[6], q[5];
cz q[14], q[9];
cz q[11], q[10];
cz q[16], q[15];
cz q[3], q[4];

h q[4];
h q[5];
h q[10];
h q[14];
h q[15];


cz q[11], q[9];
cz q[15], q[14];

h q[9];
h q[14];