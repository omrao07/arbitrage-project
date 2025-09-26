"use server";

export interface CancelOrderInput {
  orderId: string;
  accountId?: string;
  reason?: string;
}

export interface CancelOrderResult {
  success: boolean;
  orderId: string;
  accountId?: string;
  message?: string;
  error?: string;
}

/**
 * Cancels an open order.
 * Mocked for now — replace with broker API integration.
 */
export async function cancelOrder(
  input: CancelOrderInput
): Promise<CancelOrderResult> {
  const { orderId, accountId, reason } = input;

  try {
    // TODO: integrate with broker API or internal execution engine
    console.log(
      `[cancelOrder] cancelling order ${orderId} (account=${accountId}) reason=${reason}`
    );

    // mock latency
    await new Promise((r) => setTimeout(r, 250));

    // pretend it worked
    return {
      success: true,
      orderId,
      accountId,
      message: `Order ${orderId} cancelled successfully`,
    };
  } catch (err: any) {
    console.error("cancelOrder failed:", err);
    return {
      success: false,
      orderId,
      accountId,
      error: err.message || "Unknown error",
    };
  }
}