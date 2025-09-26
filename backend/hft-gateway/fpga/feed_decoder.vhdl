-- fpga/feed_decoder.vhdl
-- Simple 2-beat stream decoder: (sym_id, bid, ask, ts_low) -> packed bus
-- VHDL-2008, synthesizable

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity feed_decoder is
  generic (
    DATA_W      : integer := 64;   -- stream width
    OUT_W       : integer := 128   -- packed output width
  );
  port (
    clk                    : in  std_logic;
    rst_n                  : in  std_logic;

    -- Input stream (e.g., post-PHY / MAC)
    rx_tdata               : in  std_logic_vector(DATA_W-1 downto 0);
    rx_tvalid              : in  std_logic;
    rx_tlast               : in  std_logic;  -- asserted on beat 1 (optional, not required)
    rx_tready              : out std_logic,  -- always ready (SPSC)

    -- Packed output (for DMA or ring buffer write)
    feed_decoder_data      : out std_logic_vector(OUT_W-1 downto 0);
    feed_decoder_tvalid    : out std_logic;

    -- Optional decoded fan-out (debug/monitoring)
    dec_sym_id             : out std_logic_vector(15 downto 0);
    dec_bid_q16_16         : out std_logic_vector(31 downto 0);
    dec_ask_q16_16         : out std_logic_vector(31 downto 0);
    dec_ts_low             : out std_logic_vector(31 downto 0)
  );
end entity;

architecture rtl of feed_decoder is

  type state_t is (S_IDLE, S_GOT0);
  signal st              : state_t := S_IDLE;

  -- Beat 0 latches
  signal sym_id_r        : std_logic_vector(15 downto 0) := (others => '0');
  signal bid_r           : std_logic_vector(31 downto 0) := (others => '0');

  -- Beat 1 latches
  signal ask_r           : std_logic_vector(31 downto 0) := (others => '0');
  signal ts_low_r        : std_logic_vector(31 downto 0) := (others => '0');

  -- Output regs
  signal out_bus_r       : std_logic_vector(OUT_W-1 downto 0) := (others => '0');
  signal out_valid_r     : std_logic := '0';

begin
  -- Always ready (single-producer / single-consumer)
  rx_tready <= '1';

  -- Combinational mapping for debug taps
  dec_sym_id     <= sym_id_r;
  dec_bid_q16_16 <= bid_r;
  dec_ask_q16_16 <= ask_r;
  dec_ts_low     <= ts_low_r;

  -- Registered outputs
  feed_decoder_data   <= out_bus_r;
  feed_decoder_tvalid <= out_valid_r;

  p_fsm : process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        st          <= S_IDLE;
        sym_id_r    <= (others => '0');
        bid_r       <= (others => '0');
        ask_r       <= (others => '0');
        ts_low_r    <= (others => '0');
        out_bus_r   <= (others => '0');
        out_valid_r <= '0';
      else
        -- default: valid is a 1-cycle pulse
        out_valid_r <= '0';

        case st is
          when S_IDLE =>
            if rx_tvalid = '1' then
              -- Beat 0 fields
              sym_id_r <= rx_tdata(63 downto 48);
              bid_r    <= rx_tdata(47 downto 16);
              st       <= S_GOT0;
            end if;

          when S_GOT0 =>
            if rx_tvalid = '1' then
              -- Beat 1 fields
              ask_r    <= rx_tdata(63 downto 32);
              ts_low_r <= rx_tdata(31 downto 0);

              -- Pack into 128-bit bus:
              -- [127:112]=sym_id | [111:80]=bid | [79:48]=ask | [47:16]=ts_low | [15:0]=0
              out_bus_r <= sym_id_r & bid_r & ask_r & ts_low_r & (16 => '0', others => '0');

              out_valid_r <= '1';
              st          <= S_IDLE;
            end if;
        end case;

      end if;
    end if;
  end process;

end architecture;