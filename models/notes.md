from diffusers.models.embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps

...
self.control_add_embedding = TimestepEmbedding(addition_time_embed_dim * num_control_type, time_embed_dim)

...
control_emb = self.control_add_embedding(control_embeds) 
