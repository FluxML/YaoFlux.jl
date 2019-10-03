using Zygote
import Base.Broadcast: broadcasted
import Zygote: Context

Zygote._forward(Context(), broadcasted, sin, [0.1, 0.2])
