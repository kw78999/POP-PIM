#include "resources.h"
#include "diagnostics.h"
#include "dram_controller/controller.h"
#include "memory_system/memory_system.h"
#include <algorithm>
#include <array>
#include <climits>

namespace Ramulator {

class PopPIMHBM3Controller final : public IDRAMController, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IDRAMController, PopPIMHBM3Controller, "PopPIMHBM3", "HBM3 with Pop-PIM bank exclusion.");
  private:
    std::deque<Request> pending;          // A queue for read requests that are about to finish (callback after RL)

    ReqBuffer m_active_buffer;            // Buffer for requests being served. This has the highest priority 
    ReqBuffer m_priority_buffer;          // Buffer for high-priority requests (e.g., maintenance like refresh).
    ReqBuffer m_read_buffer;              // Read request buffer
    ReqBuffer m_write_buffer;             // Write request buffer

    int m_row_addr_idx = -1;

    float m_wr_low_watermark;
    float m_wr_high_watermark;
    bool  m_is_write_mode = false;

    std::vector<IControllerPlugin*> m_plugins;

    // ---- Refresh arbitration (strict, wc_ab, per-bank pim_pb) ---------------------------------------------
    // strict: retained behavior (a blocked priority refresh stalls every other request).
    // wc_ab: same all-bank REFab manager, but a blocked REFab only holds its own scope;
    //        requests outside that pseudochannel keep using the bus (work-conserving).
    // pim_pb: PIM-aware single-bank REFsb engine inside the controller (no REFab manager).
    int m_policy = 0;
    int64_t m_postpone = 8, m_pullin = 8, m_headroom = 2, m_drain_low = 5;
    bool m_round_rule = true;
    int m_nREFI = 0, m_refsb_cmd = -1, m_refsb_req = -1;
    std::array<uint64_t,64> m_next_due{};
    std::array<int64_t,64> m_due{}, m_issued{};
    std::array<bool,64> m_drain{}, m_eff{};
    bool m_escalate = false;
    int m_target = -1;
    ReqBuffer m_refresh_buffer;
    bool m_wc_active = false;
    AddrVec_t m_wc_scope;

    size_t s_num_row_hits = 0;
    size_t s_num_row_misses = 0;
    size_t s_num_row_conflicts = 0;


  public:
    void init() override {
      m_wr_low_watermark =  param<float>("wr_low_watermark").desc("Threshold for switching back to read mode.").default_val(0.2f);
      m_wr_high_watermark = param<float>("wr_high_watermark").desc("Threshold for switching to write mode.").default_val(0.8f);

      {
        std::string policy = param<std::string>("refresh_policy").default_val("strict");
        if (policy == "strict") m_policy = 0;
        else if (policy == "wc_ab") m_policy = 1;
        else if (policy == "pim_pb") m_policy = 2;
        else throw ConfigurationError("Unknown refresh_policy {}", policy);
        m_postpone = int(param<int>("refresh_postpone").default_val(8));
        m_pullin = int(param<int>("refresh_pullin").default_val(8));
        m_headroom = int(param<int>("refresh_headroom").default_val(2));
        m_round_rule = bool(param<bool>("refresh_round_rule").default_val(true));
        m_escalate = bool(param<bool>("refresh_laggard_escalation").default_val(false));
        m_drain_low = int(param<int>("refresh_drain_low").default_val(int(m_postpone - m_headroom - 1)));
        if (m_postpone < 1 || m_pullin < 0 || m_headroom < 1 || m_headroom > m_postpone ||
            m_drain_low > m_postpone - m_headroom - 1 || m_drain_low < -m_pullin)
          throw ConfigurationError("Invalid Phase20 refresh parameters");
        phase20diag::policy = m_policy;
        phase20diag::drain_reads = m_policy != 0;
        phase20diag::postpone = m_postpone; phase20diag::pullin = m_pullin;
        phase20diag::headroom = m_headroom; phase20diag::round_rule = m_round_rule;
      }
      m_scheduler = create_child_ifce<IScheduler>();
      m_refresh = create_child_ifce<IRefreshManager>();    

      if (m_config["plugins"]) {
        YAML::Node plugin_configs = m_config["plugins"];
        for (YAML::iterator it = plugin_configs.begin(); it != plugin_configs.end(); ++it) {
          m_plugins.push_back(create_child_ifce<IControllerPlugin>(*it));
        }
      }
    };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      m_dram = memory_system->get_ifce<IDRAM>();
      m_row_addr_idx = m_dram->m_levels("row");
      m_priority_buffer.max_size = 512*3 + 32;
      m_nREFI = m_dram->m_timing_vals("nREFI");
      phase20diag::nREFI = m_nREFI;
      m_refsb_cmd = m_dram->m_commands("REFsb");
      m_refsb_req = m_dram->m_requests("per-bank-refresh");
      // Bank b is the (b>>1)-th of 32 banks in pseudochannel (b&1). Its k-th refresh
      // obligation (k>=1) falls at (k-1)*tREFI + (idx+1)*tREFI/32: staggered and never
      // later than the all-bank schedule, whose first REFab is at tREFI.
      for (int b = 0; b < 64; b++) m_next_due[b] = uint64_t(m_nREFI) * uint64_t((b >> 1) + 1) / 32;
    };

    bool send(Request& req) override {
      req.final_command = m_dram->m_request_translations(req.type_id);

      // Forward existing write requests to incoming read requests
      if (req.type_id == Request::Type::Read) {
        auto compare_addr = [req](const Request& wreq) {
          return wreq.addr == req.addr;
        };
        if (std::find_if(m_write_buffer.begin(), m_write_buffer.end(), compare_addr) != m_write_buffer.end()) {
          // The request will depart at the next cycle
          req.depart = m_clk + 1;
          pending.push_back(req);
          return true;
        }
      }

      // Else, enqueue them to corresponding buffer based on request type id
      bool is_success = false;
      req.arrive = m_clk;
      if        (req.type_id == Request::Type::Read) {
        is_success = m_read_buffer.enqueue(req);
      } else if (req.type_id == Request::Type::Write) {
        is_success = m_write_buffer.enqueue(req);
      } else {
        throw std::runtime_error("Invalid request type!");
      }
      if (!is_success) {
        // We could not enqueue the request
        req.arrive = -1;
        return false;
      }

      return true;
    };

    bool priority_send(Request& req) override {
      req.final_command = m_dram->m_request_translations(req.type_id);

      bool is_success = false;
      is_success = m_priority_buffer.enqueue(req);
      return is_success;
    }

    void tick() override {
      m_clk++;
      poppim_hw::hw.cycle=m_clk;

      // 1. Serve completed reads
      serve_completed_reads();

      m_refresh->tick();
      phase17diag::refresh_pending.fill(false);
      phase17diag::priority_peak=std::max(phase17diag::priority_peak,uint64_t(m_priority_buffer.size()));
      for(auto& q:m_priority_buffer)for(int b=0;b<64;b++)
        if(poppim_hw::hw.targets(q.addr_vec,b))phase17diag::refresh_pending[b]=true;
      if(m_priority_buffer.size()){
        auto& q=*m_priority_buffer.begin();
        int c=m_dram->get_preq_command(q.final_command,q.addr_vec);
        if(poppim_hw::hw.blocked_by_compute(q.addr_vec))phase17diag::priority_compute_stalls++;
        else if(!m_dram->check_ready(c,q.addr_vec))phase17diag::priority_timing_stalls++;
      }


      int first_command=-1;
      if (m_policy == 2) {
        update_refresh_obligations();
        int cmd = try_refresh();
        if (cmd >= 0) {
          first_command = cmd;
          // A refresh-path row command occupies this cycle's primary slot; a column
          // command may still pair with it below exactly as for any row command.
          goto second_command;
        }
      }
      {
      // 2. Try to find a request to serve.
      ReqBuffer::iterator req_it;
      ReqBuffer* buffer = nullptr;
      bool request_found = schedule_request(req_it, buffer);

      // 3. Update all plugins
      for (auto plugin : m_plugins) {
        plugin->update(request_found, req_it);
      }

      // 4. Finally, issue the commands to serve the request
      if (request_found) {
        // If we find a real request to serve
        first_command=req_it->command;
        record_issue(req_it->command, req_it->addr_vec);
        m_dram->issue_command(req_it->command, req_it->addr_vec);

        // If we are issuing the last command, set depart clock cycle and move the request to the pending queue
        if (req_it->command == req_it->final_command) {
          if (req_it->type_id == Request::Type::Read) {
            req_it->depart = m_clk + m_dram->m_read_latency;
            pending.push_back(*req_it);
          } else if (req_it->type_id == Request::Type::Write) {
            // TODO: Add code to update statistics
          }
          buffer->remove(req_it);
        } else {
          if (m_dram->m_command_meta(req_it->command).is_opening) {
            m_active_buffer.enqueue(*req_it);
            buffer->remove(req_it);
          }
        }
      }
      }
      second_command:

      // 5. If next command is ROW/COL command while issued command is COL/ROW command, issue next command concurrently
      ReqBuffer::iterator sec_req_it;
      ReqBuffer* sec_buffer = nullptr;
      bool sec_request_found = (first_command>=0 && schedule_sec_request(sec_req_it, sec_buffer, first_command));
      for (auto plugin : m_plugins) {
        plugin->update(sec_request_found, sec_req_it);
      }
      if (sec_request_found) {
        record_issue(sec_req_it->command, sec_req_it->addr_vec);
        m_dram->issue_command(sec_req_it->command, sec_req_it->addr_vec);

        if (sec_req_it->command == sec_req_it->final_command) {
          if (sec_req_it->type_id == Request::Type::Read) {
            sec_req_it->depart = m_clk + m_dram->m_read_latency;
            pending.push_back(*sec_req_it);
          } else if (sec_req_it->type_id == Request::Type::Write) {
          }
          sec_buffer->remove(sec_req_it);
        } else {
          if (m_dram->m_command_meta(sec_req_it->command).is_opening) {
            m_active_buffer.enqueue(*sec_req_it);
            sec_buffer->remove(sec_req_it);
          }
        }
      }
    };


    bool is_pending() override {
      bool is_pending = m_active_buffer.size() ||  m_priority_buffer.size() || m_read_buffer.size() || m_write_buffer.size() || pending.size();
      return is_pending;
    };

  private:
    bool resource_ready(int command, const AddrVec_t& a) {
      if(poppim_hw::hw.blocked_by_compute(a)){poppim_hw::hw.blocked++;return false;}
      return m_dram->check_ready(command,a);
    }
    void record_issue(int command,const AddrVec_t& a){
      poppim_hw::hw.issued(command,a,m_dram->m_commands("REFab"),m_dram->m_timing_vals("nRFC"),
                           m_refsb_cmd,m_dram->m_timing_vals("nRFCSB"));
    }

    // FRFCFS (identical compare rule) restricted to requests outside a blocked refresh scope.
    ReqBuffer::iterator best_outside(ReqBuffer& buffer, const AddrVec_t& scope) {
      auto inside = [&](const Request& r) {
        for (int i = 0; i < m_row_addr_idx && i < int(scope.size()); i++)
          if (scope[i] >= 0 && scope[i] != r.addr_vec[i]) return false;
        return true;
      };
      std::vector<AddrVec_t> hits;
      for (auto& r : buffer) {
        r.command = m_dram->get_preq_command(r.final_command, r.addr_vec);
        if (m_dram->check_rowbuffer_hit(r.command, r.addr_vec))
          hits.emplace_back(r.addr_vec.begin(), r.addr_vec.begin() + m_row_addr_idx);
      }
      auto best = buffer.end();
      for (auto it = buffer.begin(); it != buffer.end(); ++it) {
        if (inside(*it)) continue;
        if (best == buffer.end()) { best = it; continue; }
        bool r1 = m_dram->check_ready(best->command, best->addr_vec);
        bool r2 = m_dram->check_ready(it->command, it->addr_vec);
        if (r1 ^ r2) {
          if (r1) continue;
          AddrVec_t bank(it->addr_vec.begin(), it->addr_vec.begin() + m_row_addr_idx);
          if (!m_dram->check_rowbuffer_hit(it->command, it->addr_vec) &&
              std::find(hits.begin(), hits.end(), bank) != hits.end()) continue;
          best = it; continue;
        }
        if (!(best->arrive <= it->arrive)) best = it;
      }
      return best;
    }

    static AddrVec_t bank_vec(int b) { return {0, b & 1, (b >> 3) & 1, (b >> 1) & 3, (b >> 4) & 3, -1, -1}; }

    void update_refresh_obligations() {
      for (int b = 0; b < 64; b++) {
        while (uint64_t(m_clk) >= m_next_due[b]) { m_due[b]++; m_next_due[b] += m_nREFI; }
        int64_t debt = m_due[b] - m_issued[b];
        if (debt > m_postpone) phase20diag::violations++;
        phase20diag::max_debt = std::max(phase20diag::max_debt, debt);
        // Hysteresis: once a bank must drain, keep draining down to the low watermark.
        if (debt >= m_postpone - m_headroom) m_drain[b] = true;
        else if (debt <= m_drain_low) m_drain[b] = false;
        phase20diag::final_debt[b] = debt; phase20diag::issued[b] = m_issued[b];
      }
      // Optional round-rule laggard escalation: if a drained bank is blocked only because the
      // round has not advanced, also drain the banks that still owe this round's refresh.
      std::array<int64_t,2> least{INT64_MAX, INT64_MAX};
      for (int b = 0; b < 64; b++) least[b & 1] = std::min(least[b & 1], m_issued[b]);
      std::array<bool,2> stuck{};
      for (int b = 0; b < 64; b++) if (m_drain[b] && m_issued[b] > least[b & 1]) stuck[b & 1] = true;
      for (int b = 0; b < 64; b++) {
        m_eff[b] = m_drain[b] || (m_escalate && m_round_rule && stuck[b & 1] && m_issued[b] == least[b & 1]);
        phase17diag::refresh_pending[b] = m_eff[b];
        if (m_eff[b]) phase20diag::urgent_bank_cycles++;
      }
    }

    // Returns the issued command id, or -1 when no refresh-path command was issued.
    int try_refresh() {
      std::array<int64_t,2> least{INT64_MAX, INT64_MAX};
      for (int b = 0; b < 64; b++) least[b & 1] = std::min(least[b & 1], m_issued[b]);
      std::array<bool,64> queued{};
      auto mark = [&](ReqBuffer& q) { for (auto& r : q) for (int b = 0; b < 64; b++) if (poppim_hw::hw.targets(r.addr_vec, b)) queued[b] = true; };
      mark(m_read_buffer); mark(m_active_buffer);
      auto eligible = [&](int b) {
        if (poppim_hw::hw.compute[b] || uint64_t(m_clk) < poppim_hw::hw.refresh_until[b]) return false;
        if (m_due[b] - m_issued[b] - 1 < -m_pullin) return false;
        if (m_round_rule && m_issued[b] > least[b & 1]) return false;
        return m_eff[b] || (poppim_hw::hw.quiet[b] && !queued[b]);
      };
      int order[64]; int n = 0;
      if (m_target >= 0 && eligible(m_target)) order[n++] = m_target;
      else m_target = -1;
      int rest[64]; int r = 0;
      for (int b = 0; b < 64; b++) if (b != m_target && eligible(b)) rest[r++] = b;
      std::sort(rest, rest + r, [&](int x, int y) {
        if (m_eff[x] != m_eff[y]) return m_eff[x] > m_eff[y];
        int64_t dx = m_due[x] - m_issued[x], dy = m_due[y] - m_issued[y];
        if (dx != dy) return dx > dy;
        return x < y;
      });
      for (int i = 0; i < r; i++) order[n++] = rest[i];
      for (int i = 0; i < n; i++) {
        int b = order[i];
        AddrVec_t a = bank_vec(b);
        int cmd = m_dram->get_preq_command(m_refsb_cmd, a);
        if (m_dram->m_command_meta(cmd).is_closing) {
          bool busy = false;
          for (auto& q : m_active_buffer) if (poppim_hw::hw.targets(q.addr_vec, b)) busy = true;
          if (busy) continue;
        }
        if (!resource_ready(cmd, a)) continue;
        Request req(a, m_refsb_req);
        req.final_command = m_refsb_cmd; req.command = cmd; req.arrive = m_clk;
        m_refresh_buffer.enqueue(req);
        auto it = m_refresh_buffer.begin();
        for (auto plugin : m_plugins) plugin->update(true, it);
        record_issue(cmd, a);
        m_dram->issue_command(cmd, a);
        m_refresh_buffer.remove(it);
        if (cmd == m_refsb_cmd) {
          if (m_eff[b]) phase20diag::refsb_urgent++; else phase20diag::refsb_quiet++;
          phase20diag::refsb++; m_issued[b]++; m_target = -1;
        } else { phase20diag::refresh_pre++; m_target = b; }
        return cmd;
      }
      return -1;
    }
    /**
     * @brief    Helper function to serve the completed read requests
     * @details
     * This function is called at the beginning of the tick() function.
     * It checks the pending queue to see if the top request has received data from DRAM.
     * If so, it finishes this request by calling its callback and poping it from the pending queue.
     */
    void serve_completed_reads() {
      if (pending.size()) {
        // Check the first pending request
        auto& req = pending[0];
        if (req.depart <= m_clk) {
          // Request received data from dram
          if (req.depart - req.arrive > 1) {
            // Check if this requests accesses the DRAM or is being forwarded.
            // TODO add the stats back
          }

          if (req.callback) {
            // If the request comes from outside (e.g., processor), call its callback
            req.callback(req);
          }
          // Finally, remove this request from the pending queue
          pending.pop_front();
        }
      };
    };


    /**
     * @brief    Checks if we need to switch to write mode
     * 
     */
    void set_write_mode() {
      if (!m_is_write_mode) {
        if ((m_write_buffer.size() > m_wr_high_watermark * m_write_buffer.max_size) || m_read_buffer.size() == 0) {
          m_is_write_mode = true;
        }
      } else {
        if ((m_write_buffer.size() < m_wr_low_watermark * m_write_buffer.max_size) && m_read_buffer.size() != 0) {
          m_is_write_mode = false;
        }
      }
    };


    /**
     * @brief    Helper function to find a request to schedule from the buffers.
     * 
     */
    bool schedule_request(ReqBuffer::iterator& req_it, ReqBuffer*& req_buffer) {
      bool request_found = false;
      // 2.1    First, check the act buffer to serve requests that are already activating (avoid useless ACTs)
      if (req_it= m_scheduler->get_best_request(m_active_buffer); req_it != m_active_buffer.end()) {
        if (resource_ready(req_it->command, req_it->addr_vec)) {
          request_found = true;
          req_buffer = &m_active_buffer;
        }
      }

      // 2.2    If no requests can be scheduled from the act buffer, check the rest of the buffers
      if (!request_found) {
        // 2.2.1    We first check the priority buffer to prioritize e.g., maintenance requests
        if (m_priority_buffer.size() != 0) {
          req_buffer = &m_priority_buffer;
          req_it = m_priority_buffer.begin();
          req_it->command = m_dram->get_preq_command(req_it->final_command, req_it->addr_vec);
          
          request_found = resource_ready(req_it->command, req_it->addr_vec);
          if (!request_found & m_priority_buffer.size() != 0) {
            if (m_policy != 1) return false;
            // Work-conserving: hold only the blocked refresh's own scope.
            m_wc_active = true; m_wc_scope = req_it->addr_vec; phase20diag::wc_bypass_cycles++;
          }
        }

        // 2.2.1    If no request to be scheduled in the priority buffer, check the read and write buffers.
        if (!request_found) {
          // Query the write policy to decide which buffer to serve
          set_write_mode();
          auto& buffer = m_is_write_mode ? m_write_buffer : m_read_buffer;
          bool wc = m_wc_active; m_wc_active = false;
          if (req_it = wc ? best_outside(buffer, m_wc_scope) : m_scheduler->get_best_request(buffer); req_it != buffer.end()) {
            request_found = resource_ready(req_it->command, req_it->addr_vec);
            req_buffer = &buffer;
          }
        }
      }

      // 2.3 If we find a request to schedule, we need to check if it will close an opened row in the active buffer.
      if (request_found) {
        if (m_dram->m_command_meta(req_it->command).is_closing) {
          std::vector<Addr_t> rowgroup((req_it->addr_vec).begin(), (req_it->addr_vec).begin() + m_row_addr_idx);

          // Search the active buffer with the row address (inkl. banks, etc.)
          for (auto _it = m_active_buffer.begin(); _it != m_active_buffer.end(); _it++) {
            std::vector<Addr_t> _it_rowgroup(_it->addr_vec.begin(), _it->addr_vec.begin() + m_row_addr_idx);
            if (rowgroup == _it_rowgroup) {
              // Invalidate this scheduling outcome if we are to interrupt a request in the active buffer
              request_found = false;
            }
          }
        }
      }

      return request_found;
    }


    /**
     * @brief    Helper function to find a second request to schedule from the buffers.
     * 
     */
    bool schedule_sec_request(ReqBuffer::iterator& req_it, ReqBuffer*& req_buffer, int first_command) {
      bool request_found = false;
      // 5.1    First, check the act buffer to serve requests that are already activating (avoid useless ACTs)
      if (req_it= m_scheduler->get_best_request(m_active_buffer); req_it != m_active_buffer.end()) {
        if (resource_ready(req_it->command, req_it->addr_vec)) {
          if (compare_command_type(first_command, req_it->command)) {
            request_found = true;
            req_buffer = &m_active_buffer;
          }
        }
      }

      // 5.2    If no requests can be scheduled from the act buffer, check the rest of the buffers
      if (!request_found) {
        // 5.2.1    We first check the priority buffer to prioritize e.g., maintenance requests
        if (m_priority_buffer.size() != 0) {
          req_buffer = &m_priority_buffer;
          req_it = m_priority_buffer.begin();
          req_it->command = m_dram->get_preq_command(req_it->final_command, req_it->addr_vec);
          
          if (compare_command_type(first_command, req_it->command)) {
            request_found = resource_ready(req_it->command, req_it->addr_vec);
            if (!request_found & m_priority_buffer.size() != 0) {
              if (m_policy != 1) return false;
              m_wc_active = true; m_wc_scope = req_it->addr_vec;
            }
          }
        }

        // 5.2.1    If no request to be scheduled in the priority buffer, check the read and write buffers.
        if (!request_found) {
          // Query the write policy to decide which buffer to serve
          set_write_mode();
          auto& buffer = m_is_write_mode ? m_write_buffer : m_read_buffer;
          bool wc = m_wc_active; m_wc_active = false;
          if (req_it = wc ? best_outside(buffer, m_wc_scope) : m_scheduler->get_best_request(buffer); req_it != buffer.end()) {
            if (compare_command_type(first_command, req_it->command)) {
              request_found = resource_ready(req_it->command, req_it->addr_vec);
              req_buffer = &buffer;
            }
          }
        }
      }

      // 5.3 If we find a request to schedule, we need to check if it will close an opened row in the active buffer.
      if (request_found) {
        if (m_dram->m_command_meta(req_it->command).is_closing) {
          std::vector<Addr_t> rowgroup((req_it->addr_vec).begin(), (req_it->addr_vec).begin() + m_row_addr_idx);

          // Search the active buffer with the row address (inkl. banks, etc.)
          for (auto _it = m_active_buffer.begin(); _it != m_active_buffer.end(); _it++) {
            std::vector<Addr_t> _it_rowgroup(_it->addr_vec.begin(), _it->addr_vec.begin() + m_row_addr_idx);
            if (rowgroup == _it_rowgroup) {
              // Invalidate this scheduling outcome if we are to interrupt a request in the active buffer
              request_found = false;
            }
          }
        }
      }

      return request_found;
    }

    // Row command: 0,  Col command: 1
    int get_command_type(int command) {
      switch (command)  {
          case 0:  return 0; // ACT,   Row command
          case 1:  return 0; // PRE,   Row command
          case 2:  return 0; // PREA,  Row command
          case 3:  return 1; // RD,    Col command
          case 4:  return 1; // WR,    Col command
          case 5:  return 1; // RDA,   Col command
          case 6:  return 1; // WRA,   Col command
          case 7:  return 0; // REFab, Row command
          case 8:  return 0; // REBsb, Row command
          case 9:  return 0; // RFMab, Row command
          case 10: return 0; // RFMsb, Row command
          default: return -1;
      }
    }

    bool compare_command_type(int first_command, int second_command) {
      int first_command_type = get_command_type(first_command);
      int second_command_type = get_command_type(second_command);
 
      if (first_command_type == -1 || second_command_type == -1) {
        return false;
      }

      if (first_command_type == second_command_type) {
        return false;
      }
      else {
        return true;
      }
    }


};
  
}   // namespace Ramulator
