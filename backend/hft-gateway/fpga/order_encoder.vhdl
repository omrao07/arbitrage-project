-- fpga/order_encoder.vhdl
-- VHDL-2008, synthesizable
-- Encodes an order into a 2-beat 64-bit AXI-Stream frame.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity order_encoder is
  generic (
    DATA_W : integer := 64
  );
  port (
    clk            : in  std_logic;
    rst_n          : in  std_logic;

    -- Order input handshake
    order_valid    : in  std_logic;
    order_ready    : out std_logic;

    -- Order fields
    sym_id         : in  std_logic_vector(15 downto 0);
    side_sell      : in  std_logic;                  -- '0' = BUY, '1' = SELL
    flags          : in  std_logic_vector(6 downto 0); -- optional flags, pack to zero if unused
    qty_q16_16     : in  std_logic_vector(31 downto 0);
    px_q16_16      : in  std_logic_vector(31 downto 0);
    ts_low         : in  std_logic_vector(31 downto 0);

    -- AXI-Stream TX (toward MAC/NIC/PCIe DMA)
    tx_tdata       : out std_logic_vector(DATA_W-1 downto 0);
    tx_tvalid      : out std_logic;
    tx_tready      : in  std_logic;
    tx_tlast       : out std_logic
  );
end entity;

architecture rtl of order_encoder is

  type state_t is (S_IDLE, S_BEAT0, S_BEAT1);
  signal st         : state_t := S_IDLE;

  -- Latched fields to survive backpressure during emission
  signal sym_id_r      : std_logic_vector(15 downto 0) := (others => '0');
  signal side_r        : std_logic := '0';
  signal flags_r       : std_logic_vector(6 downto 0) := (others => '0');
  signal qty_r         : std_logic_vector(31 downto 0) := (others => '0');
  signal px_r          : std_logic_vector(31 downto 0) := (others => '0');
  signal ts_r          : std_logic_vector(31 downto 0) := (others => '0');

  signal tdata_r       : std_logic_vector(DATA_W-1 downto 0) := (others => '0');
  signal tvalid_r      : std_logic := '0';
  signal tlast_r       : std_logic := '0';

begin
  tx_tdata  <= tdata_r;
  tx_tvalid <= tvalid_r;
  tx_tlast  <= tlast_r;

  -- Ready to accept a new order only in IDLE
  order_ready <= '1' when st = S_IDLE else '0';

  process(clk)
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        st       <= S_IDLE;
        tvalid_r <= '0';
        tlast_r  <= '0';
        tdata_r  <= (others => '0');
        sym_id_r <= (others => '0');
        side_r   <= '0';
        flags_r  <= (others => '0');
        qty_r    <= (others => '0');
        px_r     <= (others => '0');
        ts_r     <= (others => '0');

      else
        -- defaults each cycle
        if tvalid_r = '1' and tx_tready = '1' then
          -- beat consumed; clear valid unless set again below
          tvalid_r <= '0';
          tlast_r  <= '0';
        end if;

        case st is
          when S_IDLE =>
            if order_valid = '1' then
              -- Latch all fields
              sym_id_r <= sym_id;
              side_r   <= side_sell;
              flags_r  <= flags;
              qty_r    <= qty_q16_16;
              px_r     <= px_q16_16;
              ts_r     <= ts_low;

              -- Prepare BEAT 0 payload:
              -- [63:48]=sym_id, [47]=side, [46:40]=flags, [39:8]=qty, [7:0]=res0(0)
              tdata_r  <= sym_id & side_sell & flags & qty_q16_16 & (others => '0')(7 downto 0);
              tvalid_r <= '1';
              tlast_r  <= '0';
              st       <= S_BEAT0;
            end if;

          when S_BEAT0 =>
            -- Hold BEAT0 until accepted
            if tvalid_r = '1' then
              if tx_tready = '1' then
                -- After BEAT0 consumed, prepare BEAT1
                tdata_r  <= px_r & ts_r;  -- [63:32]=px, [31:0]=ts_low
                tvalid_r <= '1';
                tlast_r  <= '1';          -- second beat marks end of packet
                st       <= S_BEAT1;
              end if;
            else
              -- Defensive: if valid was cleared externally, re-drive beat 0
              tdata_r  <= sym_id_r & side_r & flags_r & qty_r & (others => '0')(7 downto 0);
              tvalid_r <= '1';
              tlast_r  <= '0';
            end if;

          when S_BEAT1 =>
            -- Hold BEAT1 until accepted
            if tvalid_r = '1' then
              if tx_tready = '1' then
                -- Packet done
                tvalid_r <= '0';
                tlast_r  <= '0';
                st       <= S_IDLE;
              end if;
            else
              -- Defensive: re-drive beat 1 if needed
              tdata_r  <= px_r & ts_r;
              tvalid_r <= '1';
              tlast_r  <= '1';
            end if;

        end case;
      end if;
    end if;
  end process;

end architecture;