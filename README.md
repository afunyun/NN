# NN

This project has lasted long enough to where I need to actually write a readme

This is primarly Iggly's NN system, where I am not using torch or any NN framework at all
I'm creating my own logic for all of the components, and we pray it works

This has two main design desisions:
- Floating point is cringe
- Error instead of Hallucinating stuff that doesn't exist

Because of the second one, I have fallen into a pattern matching **at the moment**

Core Feature list before release:
- [b] Bit Array <-> Bit Array
- [ ] ~~port to C++~~
- [ ] ~~port to Rust~~

Features I need to figure out how TF they need to work:
- Bit Array <-> Uint (Serialization flavor)
- Bit Array <-> Uint (Incremental flavor)
