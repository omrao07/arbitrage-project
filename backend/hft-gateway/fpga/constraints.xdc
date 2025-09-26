# fpga/constraints.xdc
# -------------------------------------------------------------------
# FPGA Constraints for HFT Gateway
# -------------------------------------------------------------------
# Board/device: Xilinx Ultrascale+ (example: VCU118)
# Adjust pins/locs to match your target board + NIC PHY interface.
# -------------------------------------------------------------------

# ===================== Clock Constraints ===========================
# 156.25 MHz reference clock for 10G Ethernet MAC
create_clock -name sys_clk -period 6.4 [get_ports sys_clk_p]

# Additional user clock (100 MHz)
create_clock -name user_clk -period 10.0 [get_ports user_clk]

# ===================== Input/Output Ports ==========================
# Reset pin (active low)
set_property PACKAGE_PIN AB12 [get_ports reset_n]
set_property IOSTANDARD LVCMOS18 [get_ports reset_n]

# QSFP28 transceiver reference clock pins
set_property PACKAGE_PIN Y6  [get_ports qsfp_refclk_p]
set_property PACKAGE_PIN Y5  [get_ports qsfp_refclk_n]
set_property IOSTANDARD DIFF_SSTL18 [get_ports qsfp_refclk_*]

# Ethernet RX/TX data lanes (example mapping, adjust per board)
set_property PACKAGE_PIN W10 [get_ports rxp[0]]
set_property PACKAGE_PIN W9  [get_ports rxn[0]]
set_property PACKAGE_PIN U8  [get_ports txp[0]]
set_property PACKAGE_PIN U7  [get_ports txn[0]]

# ===================== Timing Constraints ==========================
# Order encoder → NIC transmit path
set_max_delay -from [get_ports {order_encoder_data[*]}] \
              -to   [get_ports {txp[*]}] 4.0

# Feed decoder → market-data ring buffer path
set_max_delay -from [get_ports {rxp[*]}] \
              -to   [get_ports {feed_decoder_data[*]}] 4.0

# ===================== Misc Constraints ============================
# Prevent IO buffer insertion on certain streaming ports
set_property DONT_TOUCH true [get_ports {order_encoder_tvalid}]
set_property DONT_TOUCH true [get_ports {feed_decoder_tvalid}]

# Keep hierarchy for easier timing debug
set_property KEEP_HIERARCHY yes [get_cells order_encoder*]
set_property KEEP_HIERARCHY yes [get_cells feed_decoder*]