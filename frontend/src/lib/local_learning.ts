/**
 * Local Learning Module — Federated Learning edge component.
 * Captures user corrections, simulates gradient computation,
 * applies client-side DP noise, and uploads to the aggregator.
 */

const ADAPTER_DIM = 128; // Must match backend ADAPTER_DIM
const CLIP_NORM = 1.0;
const NOISE_MULTIPLIER = 1.1;

/**
 * Simulate a local gradient update based on user correction.
 * In a real system, this would be actual backprop on a local adapter.
 * Here we simulate a meaningful delta vector.
 */
function computeSimulatedGradient(
    originalResponse: string,
    userCorrection: string
): number[] {
    // Simulate gradient as a hash-derived vector (placeholder for real training)
    const combined = originalResponse + userCorrection;
    const gradients: number[] = [];
    for (let i = 0; i < ADAPTER_DIM; i++) {
        const charCode = combined.charCodeAt(i % combined.length) || 0;
        gradients.push((charCode / 127.5 - 1.0) * 0.1); // small values near 0
    }
    return gradients;
}

/**
 * Client-side gradient clipping (L2 norm clipping).
 */
function clipGradients(gradients: number[], clipNorm: number): number[] {
    const norm = Math.sqrt(gradients.reduce((sum, g) => sum + g * g, 0));
    if (norm > clipNorm) {
        return gradients.map((g) => g * (clipNorm / norm));
    }
    return gradients;
}

/**
 * Add Gaussian noise for differential privacy.
 */
function addGaussianNoise(gradients: number[], noiseStd: number): number[] {
    return gradients.map((g) => {
        // Box-Muller transform for Gaussian noise
        const u1 = Math.random();
        const u2 = Math.random();
        const noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * noiseStd;
        return g + noise;
    });
}

/**
 * Full client-side DP pipeline: compute → clip → noise.
 */
function applyClientDP(gradients: number[]): number[] {
    const clipped = clipGradients(gradients, CLIP_NORM);
    const noiseStd = NOISE_MULTIPLIER * CLIP_NORM;
    return addGaussianNoise(clipped, noiseStd);
}

/**
 * Upload a privatized update to the backend aggregator.
 */
export async function submitFederatedUpdate(
    clientId: string,
    originalResponse: string,
    userCorrection: string,
    backendUrl: string = "http://localhost:8080"
): Promise<{ success: boolean; message: string }> {
    try {
        const rawGradients = computeSimulatedGradient(originalResponse, userCorrection);
        const dpGradients = applyClientDP(rawGradients);

        const response = await fetch(`${backendUrl}/api/federated/update`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                client_id: clientId,
                gradients: dpGradients,
            }),
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        return { success: true, message: data.message || "Update submitted." };
    } catch (err) {
        return { success: false, message: `Failed to submit update: ${err}` };
    }
}

/**
 * Download the latest global adapter from the aggregator.
 */
export async function downloadLatestAdapter(
    backendUrl: string = "http://localhost:8080"
): Promise<number[] | null> {
    try {
        const response = await fetch(`${backendUrl}/api/federated/adapter`);
        const data = await response.json();
        if (data.status === "no_adapter") return null;
        return data.adapter as number[];
    } catch {
        return null;
    }
}
