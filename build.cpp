#include <cbuild/cbuild.hpp>

int build(CBuild::Context context) {
    
    CBuild::Executable test (
        context,
        "./test.cpp",
        "test"
    );

    test.includeDirectory("include");
    test.compile();
    
    return 0;
}

CBUILD_RUN int test() {
    
    return system("./build/test");
}