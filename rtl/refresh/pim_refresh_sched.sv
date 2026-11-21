// PIM-aware single-bank refresh decision logic (64 banks = 2 pseudochannels x 32).
// Mirrors the controller policy "pim_pb": staggered per-bank obligations, signed debt window
// [-U, +P], drain hysteresis, optional round rule, sticky target, priority = drain > debt > low id.
// DRAM timing readiness is an input (ready[b]), exactly like any controller's command check.
// Verilog-2005 style so that the same file is used by Verilator 4.028 and Yosys 0.9.
module pim_refresh_sched #(
  parameter NREFI = 5070,
  parameter P = 8,
  parameter U = 8,
  parameter H = 2,
  parameter LOW = 5,
  parameter ROUND = 1
) (
  input  wire        clk,
  input  wire        rst,
  input  wire [63:0] compute,     // scorer occupancy (DRAM commands excluded)
  input  wire [63:0] refreshing,  // bank inside its own tRFCsb
  input  wire [63:0] quiet,       // PIM hint: waiting on the shared backend
  input  wire [63:0] queued,      // read/active-buffer requests target the bank
  input  wire [63:0] ready,       // next refresh-path command (PRE or REFsb) is timing-ready
  input  wire [63:0] bank_open,   // the refresh-path command is PRE when the bank is open
  output wire [63:0] drain,       // to the near-bank PIM controllers
  output wire        req_valid,   // issue this cycle
  output wire [5:0]  req_bank,
  output wire        req_is_pre,
  output reg         violation
);
  localparam CW = 13;             // ceil(log2(5071)); NREFI must be < 8192
  localparam signed [5:0] HI_T  = P - H;
  localparam signed [5:0] LO_T  = LOW;
  localparam signed [5:0] PMAX  = P;
  localparam signed [5:0] UMIN  = -U;

  reg  [CW-1:0] phase;            // m_clk mod NREFI (m_clk starts at 1)
  reg  [6*64-1:0] debt_q;
  reg  [63:0] drain_q, round_done;
  reg  [6:0]  target;             // {valid, bank}

  wire [6*64-1:0] debt_due;
  wire [63:0] drain_now, elig, cand, over;
  wire [13*64-1:0] key;

  genvar g;
  generate for (g = 0; g < 64; g = g + 1) begin : bank
    localparam [CW-1:0] DUE = ((g / 2 + 1) * NREFI / 32) % NREFI;
    localparam [5:0] IDX = g;
    wire signed [5:0] dq = debt_q[6*g +: 6];
    wire signed [5:0] dd = dq + ((phase == DUE) ? 6'sd1 : 6'sd0);
    assign debt_due[6*g +: 6] = dd;
    assign drain_now[g] = (dd >= HI_T) ? 1'b1 : (dd <= LO_T) ? 1'b0 : drain_q[g];
    wire signed [5:0] after = dd - 6'sd1;
    assign elig[g] = !compute[g] && !refreshing[g] && (after >= UMIN) &&
                     (ROUND == 0 || !round_done[g]) && (drain_now[g] || (quiet[g] && !queued[g]));
    assign cand[g] = elig[g] && ready[g];
    assign over[g] = dd > PMAX;
    wire [5:0] biased = dd + 6'sd8;
    assign key[13*g +: 13] = {cand[g], drain_now[g], biased[4:0], ~IDX};
  end endgenerate

  // 64-leaf arg-max tree; keys are unique (they embed the bank index).
  wire [13*127-1:0] node;
  generate
    for (g = 0; g < 64; g = g + 1) begin : leaf
      assign node[13*(63+g) +: 13] = key[13*g +: 13];
    end
    for (g = 62; g >= 0; g = g - 1) begin : tree
      wire [12:0] l = node[13*(2*g+1) +: 13];
      wire [12:0] r = node[13*(2*g+2) +: 13];
      assign node[13*g +: 13] = (l > r) ? l : r;
    end
  endgenerate
  wire [12:0] best = node[12:0];
  wire [5:0] best_bank = ~best[5:0];
  wire use_target = target[6] && cand[target[5:0]];
  assign req_valid  = use_target || best[12];
  assign req_bank   = use_target ? target[5:0] : best_bank;
  assign req_is_pre = bank_open[req_bank];
  assign drain = drain_now;

  wire fire_ref = req_valid && !req_is_pre;
  wire [63:0] done_set = round_done | (fire_ref ? (64'd1 << req_bank) : 64'd0);
  wire all0 = &{done_set[62], done_set[60], done_set[58], done_set[56], done_set[54], done_set[52], done_set[50], done_set[48],
               done_set[46], done_set[44], done_set[42], done_set[40], done_set[38], done_set[36], done_set[34], done_set[32],
               done_set[30], done_set[28], done_set[26], done_set[24], done_set[22], done_set[20], done_set[18], done_set[16],
               done_set[14], done_set[12], done_set[10], done_set[8],  done_set[6],  done_set[4],  done_set[2],  done_set[0]};
  wire all1 = &{done_set[63], done_set[61], done_set[59], done_set[57], done_set[55], done_set[53], done_set[51], done_set[49],
               done_set[47], done_set[45], done_set[43], done_set[41], done_set[39], done_set[37], done_set[35], done_set[33],
               done_set[31], done_set[29], done_set[27], done_set[25], done_set[23], done_set[21], done_set[19], done_set[17],
               done_set[15], done_set[13], done_set[11], done_set[9],  done_set[7],  done_set[5],  done_set[3],  done_set[1]};
  wire [63:0] even = 64'h5555555555555555;
  wire [63:0] done_next = done_set & ~(all0 ? even : 64'd0) & ~(all1 ? ~even : 64'd0);

  integer b;
  always @(posedge clk) begin
    if (rst) begin
      phase <= 1; drain_q <= 64'd0; round_done <= 64'd0; target <= 7'd0; violation <= 1'b0; debt_q <= 0;
    end else begin
      phase <= (phase == NREFI - 1) ? {CW{1'b0}} : phase + 1'b1;
      drain_q <= drain_now;
      for (b = 0; b < 64; b = b + 1)
        debt_q[6*b +: 6] <= debt_due[6*b +: 6] - ((fire_ref && req_bank == b) ? 6'd1 : 6'd0);
      if (|over) violation <= 1'b1;
      // Sticky target: cleared when no longer eligible or after REFsb; set after a refresh PRE.
      if (req_valid) target <= req_is_pre ? {1'b1, req_bank} : 7'd0;
      else if (target[6] && !elig[target[5:0]]) target <= 7'd0;
      if (ROUND != 0 && fire_ref) round_done <= done_next;
    end
  end
endmodule
