#include <vector>
#include <iomanip>

#include <gmpxx.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "../include/poly_ops/large_ints.hpp"


template<unsigned int N,bool Signed> mpz_class to_gmp(const poly_ops_new::compound_xint<N,Signed> &x) {
    mpz_class r;
    if constexpr(Signed) {
        if(x.negative()) {
            mpz_import(r.get_mpz_t(),N,-1,sizeof(poly_ops_new::full_int),0,0,(-x).data());
            return -r;
        }
    }

    mpz_import(r.get_mpz_t(),N,-1,sizeof(poly_ops_new::full_int),0,0,x.data());
    return r;
}

template<unsigned int N,bool Signed> poly_ops_new::compound_xint<N,Signed> from_gmp(const mpz_class &x) {
    using namespace poly_ops_new;

    if(Signed && sgn(x) < 0) throw std::logic_error("cannot store negative value in unsigned class");

    compound_xint<N,Signed> r{0};
    std::size_t bitsize = sizeof(full_int)*8;
    std::size_t words = (mpz_sizeinbase(x.get_mpz_t(),2) + bitsize - 1 + Signed) / bitsize;
    if(words > N) throw std::logic_error("input value is too big");
    mpz_export(r.data(),nullptr,-1,sizeof(full_int),0,0,x.get_mpz_t());
    if constexpr(Signed) {
        if(sgn(x)) return -r;
    }
    return r;
}

namespace poly_ops_new {
    namespace detail {
        template<unsigned int N> std::ostream &output_tail(std::ostream &os,const_compound_xint_ref<N> x) {
            os << ',' << std::setw(sizeof(full_int)*2) << x.hi();
            if constexpr(N > 1) {
                return output_tail(os,x.lo());
            } else {
                return os;
            }
        };
        template<unsigned int N,bool Signed> std::ostream &output_head(std::ostream &os,const compound_xint<N,Signed> &x) {
            if constexpr(N > 1) {
                return output_tail(os << x.hi(),x.lo());
            } else {
                return os << x.hi();
            }
        };
    }

    template<unsigned int N,bool Signed> std::ostream &operator<<(std::ostream &os,const compound_xint<N,Signed> &x) {
        if constexpr(Signed) {
            if(x.negative()) {
                return detail::output_head(os << "-0x" << std::hex << std::setfill('0'),-x);
            }
        }
        return detail::output_head(os << "0x" << std::hex << std::setfill('0'),x);
        //return os << to_gmp(x);
    };
}

template<unsigned int N,bool Signed> bool operator==(const poly_ops_new::compound_xint<N,Signed> &a,const mpz_class &b) {
    return to_gmp(a) == b;
}

constexpr inline poly_ops_new::full_uint ONES = ~poly_ops_new::full_uint(0);

TEST_CASE("Test compare","[compare]") {
    using namespace poly_ops_new;

    REQUIRE(compound_int<1>(0) == full_int(0));
    REQUIRE(compound_int<1>(1) != full_int(0));
    REQUIRE(compound_int<2>(0) == full_int(0));
    REQUIRE(compound_int<2>(1) != full_int(0));
    REQUIRE(compound_int<2>(1,0u) != full_int(0));
    REQUIRE(0xffffffffffffffff_compi != -1);
    REQUIRE(compound_int<1>(ONES) != compound_uint<1>(ONES));
    REQUIRE(compound_int<2>(0,ONES) == compound_uint<1>(ONES));
    REQUIRE(0xfedcba9876543210fedcba9876543210_compi == 0xfedcba9876543210fedcba9876543210_compi);
    REQUIRE(0xfedcba9876543210fedcba9876543210_compi <= 0xfedcba9876543210fedcba9876543210_compi);
    REQUIRE(0xfedcba9876543210fedcba9876543210_compi >= 0xfedcba9876543210fedcba9876543210_compi);
    REQUIRE(0xfedcba9876543210fedcba9876543210_compi < 0xfedcba9876543210fedcba9876543211_compi);
    REQUIRE(-0xfedcba9876543210fedcba9876543210_compi < 0xfedcba9876543210fedcba9876543210_compi);
    REQUIRE(-0xfedcba9876543210fedcba9876543210_compi <= 0xfedcba9876543210fedcba9876543210_compi);
    REQUIRE(0xfedcba9876543210fedcba9876543210_compi > -0xfedcba9876543210fedcba9876543210_compi);
    REQUIRE(0xfedcba9876543210fedcba9876543210_compi >= -0xfedcba9876543210fedcba9876543210_compi);
    REQUIRE(compound_int<2>(-3751) < compound_int<2>(-1260));
}

TEST_CASE("Test add","[add]") {
    using namespace poly_ops_new;

    REQUIRE(compound_int<1>(0) + compound_int<1>(0) == compound_int<1>(0));
    REQUIRE(compound_int<1>(1) + compound_int<1>(0) == compound_int<1>(1));
    REQUIRE(compound_int<1>(4) + compound_int<1>(-8) == compound_int<1>(-4));
    REQUIRE(compound_int<2>(0,ONES) + full_int(1) == compound_int<2>(1,0u));
    REQUIRE(compound_int<2>(ONES,ONES) + full_int(1) == compound_int<2>(0,0u));
    REQUIRE(compound_int<2>(1,0u) + full_int(-1) == compound_int<2>(0,ONES));
    REQUIRE(0xfedcba9876543210_compi + 0x1122334455667788_compi == 0x10ffeeddccbbaa998_compi);
    REQUIRE(compound_uint<1>(ONES) + compound_uint<2>(1,0u) == compound_uint<2>(1,ONES));
    REQUIRE(compound_int<1>(ONES) + compound_uint<2>(1,0u) == compound_int<2>(0,ONES));
}

TEST_CASE("Test subtract","[sub]") {
    using namespace poly_ops_new;

    REQUIRE(compound_int<1>(0) - compound_int<1>(0) == compound_int<1>(0));
    REQUIRE(compound_int<1>(1) - compound_int<1>(0) == compound_int<1>(1));
    REQUIRE(compound_int<1>(4) - compound_int<1>(-8) == compound_int<1>(12));
    REQUIRE(compound_int<2>(0,ONES) - full_int(-1) == compound_int<2>(1,0u));
    REQUIRE(compound_int<2>(ONES,ONES) - full_int(-1) == compound_int<2>(0,0u));
    REQUIRE(compound_int<2>(1,0u) - full_int(1) == compound_int<2>(0,ONES));
    REQUIRE(0xfedcba9876543210_compi - 0x1122334455667788_compi == 0xedba875420edba88_compi);
}

TEST_CASE("Test multiply","[multiply]") {
    using namespace poly_ops_new;

    REQUIRE(mul(full_int(5),full_int(5)) == 25);
    REQUIRE(0xffffffff_compi * 5 == 0x4fffffffb_compi);
    REQUIRE(0xfffffffffffffff_compi * 5 == 0x4ffffffffffffffb_compi);
    REQUIRE(0xfffffffffffffff_compi * -9 == -0x8ffffffffffffff7_compi);
    REQUIRE(-0xfffffffffffffff_compi * 9 == -0x8ffffffffffffff7_compi);
    REQUIRE(0xffffffffffffffff_compi * 5 == 0x4fffffffffffffffb_compi);
    REQUIRE(0xffffffffffffffff_compi * -9 == -0x8fffffffffffffff7_compi);
    REQUIRE(0xfedcba9876543210fedcba9876543210_compi * 0xabc == 0xaafc962fc962fc96e6fc962fc962fc963c0_compi);
    REQUIRE(0xfedcba9876543210fedcba9876543210_compi * -0xabc == -0xaafc962fc962fc96e6fc962fc962fc963c0_compi);
    REQUIRE(-0xfedcba9876543210fedcba9876543210_compi * 0xabc == -0xaafc962fc962fc96e6fc962fc962fc963c0_compi);
    REQUIRE(-0xfedcba9876543210fedcba9876543210_compi * -0xabc == 0xaafc962fc962fc96e6fc962fc962fc963c0_compi);
    REQUIRE(0xfedcba9876543210fedcba9876543210_compi * 0x123456789abcdef0123456789abcdef_compi == 0x121fa00ad77d742247acc9140513b74458fab20783af1222236d88fe5618cf0_compi);
}

TEST_CASE("Test shift","[shift]") {
    using namespace poly_ops_new;

    REQUIRE((0xffffffff_compi >> 1) == 0x7fffffff_compi);
    REQUIRE((0xfffffffffffffff_compi >> 6) == 0x3fffffffffffff_compi);
    REQUIRE((0x00fffffffffffffff_compi << 6) == 0x3ffffffffffffffc0_compi);
    REQUIRE((0xfedcba9876543210fedcba9876543210_compi >> 13) == 0x7f6e5d4c3b2a19087f6e5d4c3b2a1_compi);
    REQUIRE((0xfedcba9876543210fedcba9876543210_compi >> 0) == 0xfedcba9876543210fedcba9876543210_compi);
    REQUIRE((0xfedcba9876543210fedcba9876543210_compi << 0) == 0xfedcba9876543210fedcba9876543210_compi);
    REQUIRE((0x0000fedcba9876543210fedcba9876543210_compi << 13) == 0x1fdb97530eca86421fdb97530eca86420000_compi);
}

TEST_CASE("Test divide","[divide]") {
    using namespace poly_ops_new;

    REQUIRE(0xffffffff_compi / 11 == 0x1745d174_compi);
    REQUIRE(0xfffffffffffffff_compi / 12 == 0x155555555555555_compi);
    REQUIRE(0xfedcba9876543210fedcba9876543210_compi / 13 == 0x139ad346ce067a014eae8481e1b7b514_compi);
    REQUIRE(0xfedcba9876543210fedcba9876543210_compi / -13 == -0x139ad346ce067a014eae8481e1b7b514_compi);
    REQUIRE(-0xfedcba9876543210fedcba9876543210_compi / 13 == -0x139ad346ce067a014eae8481e1b7b514_compi);
    REQUIRE(-0xfedcba9876543210fedcba9876543210_compi / -13 == 0x139ad346ce067a014eae8481e1b7b514_compi);
    REQUIRE(divmod(0xfedcba9876543210fedcba9876543210_compi,13,modulo_t::euclid).quot == 0x139ad346ce067a014eae8481e1b7b514_compi);
    REQUIRE(divmod(0xfedcba9876543210fedcba9876543210_compi,-13,modulo_t::euclid).quot == -0x139ad346ce067a014eae8481e1b7b515_compi);
    REQUIRE(divmod(-0xfedcba9876543210fedcba9876543210_compi,13,modulo_t::euclid).quot == -0x139ad346ce067a014eae8481e1b7b515_compi);
    REQUIRE(divmod(-0xfedcba9876543210fedcba9876543210_compi,-13,modulo_t::euclid).quot == 0x139ad346ce067a014eae8481e1b7b514_compi);
    REQUIRE(0x121fa00ad77d742247acc9140513b74458fab20783af1222236d88fe5618cf0_compi / 0x123456789abcdef0123456789abcdef_compi == 0xfedcba9876543210fedcba9876543210_compi);
    REQUIRE(0x121fa00ad77d742247acc9140513b74458fab20783af1222236d88fe5618cf0_compi / 0xfedcba9876543210fedcba9876543210_compi == 0x123456789abcdef0123456789abcdef_compi);
    REQUIRE(0x1cca5d15d93cf3491bb81cfa68083b2d9ec8581cc3523a4c9_compi / 0x1a1c47c53b3211c279541261e94b415bdf18fc584762d7a72_compi == 1);
}

TEST_CASE("Test modulo","[modulo]") {
    using namespace poly_ops_new;

    REQUIRE(divmod(0xfedcba9876543210fedcba9876543210_compi,13).rem == 12);
    REQUIRE(divmod(0xfedcba9876543210fedcba9876543210_compi,-13).rem == 12);
    REQUIRE(divmod(-0xfedcba9876543210fedcba9876543210_compi,13).rem == -12);
    REQUIRE(divmod(-0xfedcba9876543210fedcba9876543210_compi,-13).rem == -12);
    REQUIRE(divmod(0xfedcba9876543210fedcba9876543210_compi,13,modulo_t::euclid).rem == 12);
    REQUIRE(divmod(0xfedcba9876543210fedcba9876543210_compi,-13,modulo_t::euclid).rem == 1);
    REQUIRE(divmod(-0xfedcba9876543210fedcba9876543210_compi,13,modulo_t::euclid).rem == 1);
    REQUIRE(divmod(-0xfedcba9876543210fedcba9876543210_compi,-13,modulo_t::euclid).rem == 12);
    REQUIRE(divmod(27,10,modulo_t::euclid).rem == 7);
    REQUIRE(divmod(27,-10,modulo_t::euclid).rem == 3);
    REQUIRE(divmod(-27,10,modulo_t::euclid).rem == 3);
    REQUIRE(divmod(-27,-10,modulo_t::euclid).rem == 7);
    REQUIRE(unmul<1>(27,10,modulo_t::euclid).rem == 7);
    REQUIRE(unmul<1>(27,-10,modulo_t::euclid).rem == 3);
    REQUIRE(unmul<1>(-27,10,modulo_t::euclid).rem == 3);
    REQUIRE(unmul<1>(-27,-10,modulo_t::euclid).rem == 7);
}

TEST_CASE("Ops on random numbers","[random]") {
    using namespace poly_ops_new;

    auto values = GENERATE(chunk(6,take(6000,random(full_uint(0),~full_uint(0)))));
    compound_int<4> a, b;
    a[0] = values[0];
    a[1] = values[1];
    a[2] = values[2];
    a[3] = values[2] >> (sizeof(full_uint)*8-1);
    b[0] = values[3];
    b[1] = values[4];
    b[2] = values[5];
    b[3] = values[5] >> (sizeof(full_uint)*8-1);
    auto a_gmp = to_gmp(a);
    auto b_gmp = to_gmp(b);

    CAPTURE(a,b);

    REQUIRE(a + b == a_gmp + b_gmp);
    REQUIRE(a - b == a_gmp - b_gmp);
    REQUIRE(a * b == a_gmp * b_gmp);
    if(b) REQUIRE(a / b == a_gmp / b_gmp);
    REQUIRE((a << 3) == (a_gmp << 3));
    REQUIRE((a >> 100) == (a_gmp >> 100));
    REQUIRE((a < b) == (a_gmp < b_gmp));
    REQUIRE((a <= b) == (a_gmp <= b_gmp));
    REQUIRE((a > b) == (a_gmp > b_gmp));
    REQUIRE((a >= b) == (a_gmp >= b_gmp));
    REQUIRE((a == b) == (a_gmp == b_gmp));
    REQUIRE((a != b) == (a_gmp != b_gmp));
    REQUIRE(-a == -a_gmp);
    REQUIRE(a + 1 != a_gmp + 2);
}
